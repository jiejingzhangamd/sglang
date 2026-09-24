import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _maximum(a, b):
    return gl.maximum(a, b)

@gluon.jit
def _sigmoid(x):
    return 1.0 / (1.0 + gl.exp(-x))

@gluon.jit
def _scale_offset(n, group, K: gl.constexpr):
    groups: gl.constexpr = gl.cdiv(K // 32, 8) * 8
    tile = n // 32 * (groups // 8) + group // 8
    return (((tile * 4 + group % 4) * 16 + n % 16) * 2 + group // 4 % 2) * 2 + n // 16 % 2

@gluon.jit(do_not_specialize=['BLOCKS'])
def _grouped_tile(pid, BLOCKS, COLUMNS: gl.constexpr, GROUP: gl.constexpr):
    pid = pid.to(gl.uint32)
    BLOCKS = BLOCKS.to(gl.uint32)
    first = pid // (GROUP * COLUMNS) * GROUP
    block = first + pid % GROUP
    column = pid % (GROUP * COLUMNS) // GROUP
    if GROUP != 1:
        full_blocks = BLOCKS // GROUP * GROUP
        full_tiles = full_blocks * COLUMNS
        tail_rows = BLOCKS % GROUP
        if tail_rows != 0:
            if pid >= full_tiles:
                tail = pid - full_tiles
                block = full_blocks + tail % tail_rows
                column = tail // tail_rows
    return (block.to(gl.int32), column.to(gl.int32))

@gluon.jit
def _split_columns(tile):
    return gl.split(tile.reshape((tile.shape[0], 2, tile.shape[1] // 2)).permute((0, 2, 1)))

@gluon.jit(do_not_specialize=['M'])
def _router_projection_tile(X, W, Y, row_tile, col_tile, M, K: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, 4], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    rows = row_tile * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    cols = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    a = gl.load(X + rows[:, None] * SX + ak[None, :], rows[:, None] < M, 0)
    b = gl.load(W + cols[None, :] * K + bk[:, None])
    for k in range(K // BK - 1):
        aa = gl.convert_layout(a, ad)
        bb = gl.convert_layout(b, bd)
        next_k = k + 1
        a = gl.load(X + rows[:, None] * SX + next_k * BK + ak[None, :], rows[:, None] < M, 0)
        b = gl.load(W + cols[None, :] * K + next_k * BK + bk[:, None])
        acc = gl.amd.cdna4.mfma(aa, bb, acc)
    acc = gl.amd.cdna4.mfma(gl.convert_layout(a, ad), gl.convert_layout(b, bd), acc)
    rm = row_tile * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    cn = col_tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(Y + rm[:, None] * 256 + cn[None, :], acc, rm[:, None] < M)

@gluon.jit
def _router(Logits, Bias, Ids, Weights):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    probability = _sigmoid(gl.load(Logits + m * 256 + e).to(gl.float32))
    score = probability + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected = gl.full((256,), 0.0, gl.float32, layout)
    total = 0.0
    for j in range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        if idx >= 256:
            idx = gl.min(gl.where(available, e, 256), 0)
        index = gl.full((1,), idx, gl.int32, layout)
        prob = gl.sum(gl.gather(probability, index, 0), 0)
        gl.store(Ids + m * 8 + j, idx)
        selected = gl.where(e == j, prob, selected)
        total += prob
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
    gl.store(Weights + m * 8 + e, selected / total * 2.5, e < 8)

@gluon.jit(do_not_specialize=['ROUTES', 'CHUNKS', 'CAPACITY'])
def _chunk_counts(Ids, Counts, Sorted, Experts, ROUTES, CHUNKS, CAPACITY, BM: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    chunk = gl.program_id(0)
    r = chunk * 256 + gl.arange(0, 256, layout)
    e = gl.load(Ids + r, r < ROUTES, 257)
    histogram = gl.histogram(e, 256, mask=e < 256, layout=layout)
    expert = gl.arange(0, 256, layout)
    gl.store(Counts + expert * CHUNKS + chunk, histogram, chunk < CHUNKS)
    r = chunk * 1024 + gl.arange(0, 1024, layout)
    gl.store(Sorted + r, -1, r < CAPACITY)
    gl.store(Experts + r, -1, r < CAPACITY // BM)

@gluon.jit(do_not_specialize=['CHUNKS', 'M'])
def _chunk_prefix(Counts, Prefix, Totals, CHUNKS, BLOCK: gl.constexpr, M):
    e = gl.program_id(0)
    if e < 256:
        c = gl.arange(0, BLOCK, gl.BlockedLayout([1], [64], [4], [0]))
        counts = gl.load(Counts + e * CHUNKS + c, c < CHUNKS, 0)
        prefix = gl.associative_scan(counts, 0, _add) - counts
        gl.store(Prefix + e * CHUNKS + c, prefix, c < CHUNKS)
        gl.store(Totals + e, gl.sum(counts, 0))
    else:
        gl.store(Totals + e, M)

@gluon.jit
def _build_expert_blocks(Counts, Offsets, Experts, BM: gl.constexpr, BLOCK: gl.constexpr):
    e = gl.program_id(0)
    count = gl.load(Counts + e)
    all_e = gl.arange(0, 512, gl.BlockedLayout([1], [64], [4], [0]))
    all_counts = gl.load(Counts + all_e, all_e < 257, 0)
    offset = gl.sum(gl.where(all_e < e, gl.cdiv(all_counts, BM), 0), 0)
    gl.store(Offsets + e, offset * BM)
    compact = gl.sum(gl.where(all_e < e, all_counts, 0), 0)
    gl.store(Offsets + 257 + e, compact - offset * BM)
    i = gl.arange(0, BLOCK, gl.BlockedLayout([1], [64], [4], [0]))
    rows = gl.minimum(count - i * BM, BM)
    descriptor = (compact + i * BM).to(gl.int64) << 17 | e + rows * 512
    gl.store(Experts + offset + i, descriptor, i < gl.cdiv(count, BM))

@gluon.jit(do_not_specialize=['ROUTES', 'CHUNKS'])
def _scatter(Ids, Offsets, Prefix, Sorted, Inverse, ROUTES, CHUNKS):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    chunk = gl.program_id(0)
    lane = gl.arange(0, 256, layout)
    r = chunk * 256 + lane
    e = gl.load(Ids + r, r < ROUTES, 257)
    keys = e * 256 + lane
    for stage in gl.static_range(1, 9):
        for step in gl.static_range(stage):
            distance = 1 << stage - 1 - step
            other = gl.gather(keys, lane ^ distance, 0)
            take_min = (lane & 1 << stage == 0) == (lane & distance == 0)
            keys = gl.where(take_min, gl.minimum(keys, other), gl.maximum(keys, other))
    expert = keys // 256
    previous = gl.gather(expert, gl.maximum(lane - 1, 0), 0)
    starts = gl.where((lane == 0) | (expert != previous), lane, 0)
    starts = gl.associative_scan(starts, 0, _maximum)
    route = chunk * 256 + keys % 256
    valid = (expert < 256) & (route < ROUTES)
    offset = gl.load(Offsets + expert, valid, 0)
    prefix = gl.load(Prefix + expert * CHUNKS + chunk, valid, 0)
    sorted_row = offset + prefix + lane - starts
    gl.store(Sorted + sorted_row, route, valid)
    delta = gl.load(Offsets + 257 + expert, valid, 0)
    gl.store(Inverse + route, sorted_row + delta, valid)

@gluon.jit
def _quantize_values(x, shared):
    peak = gl.max(gl.abs(x), 1)
    divided = gl.div_rn(peak, 6.0)
    bits = divided.to(gl.uint32, bitcast=True)
    exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
    peak_bits = peak.to(gl.uint32, bitcast=True)
    floor_exp = (peak_bits >> 23 & 255).to(gl.int32) - 127
    threshold = gl.exp2(floor_exp.to(gl.float32)) * 1.75
    even_exp = floor_exp - 2 + (peak >= threshold).to(gl.int32)
    exponent = gl.where(shared, even_exp, exponent)
    exponent = gl.maximum(-127, gl.minimum(127, exponent))
    scale = gl.exp2(exponent.to(gl.float32))
    inverse = gl.div_rn(1.0, scale)
    a = gl.abs(x * inverse[:, None])
    a = gl.where(a == a, a, float('inf'))
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    unit = gl.full(low.shape, 1.0, gl.float32, low.type.layout)
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3', constraints='=v,v,v,v', args=[low, high, unit], dtype=gl.uint32, is_pure=True, pack=1).to(gl.uint8)
    signs = gl.where(x < 0, 8, 0).to(gl.uint8)
    sign_low, sign_high = gl.split(signs.reshape((x.shape[0], 16, 2)))
    return (packed | sign_low | sign_high << 4, (exponent + 127).to(gl.uint8))

@gluon.jit(do_not_specialize=['M'])
def _quantize_input_tile(X, Q, QScale, tile, M, K: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, VALUES: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, VALUES], [2 * VALUES, 32 // VALUES], [4, 1], [1, 0])
    group = tile * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, layout))
    k = gl.arange(0, 32, gl.SliceLayout(0, layout))
    row = group // (K // 32)
    col = (group % (K // 32))[:, None] * 32 + k[None, :]
    x = gl.load(X + row[:, None] * SX + col, row[:, None] < M, 0).to(gl.float32)
    routed, routed_scale = _quantize_values(x, False)
    shared, shared_scale = _quantize_values(x, True)
    packed_layout: gl.constexpr = routed.type.layout
    r = gl.convert_layout(row, gl.SliceLayout(1, packed_layout))
    g = gl.convert_layout(group, gl.SliceLayout(1, packed_layout))
    b = gl.arange(0, 16, gl.SliceLayout(0, packed_layout))
    dest = g[:, None] * 16 + b[None, :]
    gl.store(Q + dest, routed, r[:, None] < M)
    gl.store(Q + M * (K // 2) + dest, shared, r[:, None] < M)
    gl.store(QScale + group, routed_scale, row < M)
    gl.store(QScale + M * (K // 32) + group, shared_scale, row < M)

@gluon.jit(do_not_specialize=['M'])
def _router_project_quantize(X, W, Logits, Q, QScale, M, K: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr, GROUPS: gl.constexpr, VALUES: gl.constexpr):
    tile = gl.program_id(0)
    rows = gl.cdiv(M, BM)
    projection_tiles = rows * (256 // BN)
    if tile < projection_tiles:
        _router_projection_tile(X, W, Logits, tile % rows, tile // rows, M, K, SX, BM, BK, BN)
    else:
        _quantize_input_tile(X, Q, QScale, tile - projection_tiles, M, K, SX, GROUPS, VALUES)

@gluon.jit
def _store_w13_activation(acc, Y, YScale, expert, start_row, column, N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr):
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    gate, up = _split_columns(acc)
    if expert == 256:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    activated = (gate * _sigmoid(gate) * up).to(gl.bfloat16)
    activated = gl.convert_layout(activated, ep).reshape((BM * BN // 64, 32)).to(gl.float32)
    packed, scale = _quantize_values(activated, expert == 256)
    packed = gl.convert_layout(packed.reshape((BM, BN // 4)), ep)
    scale = gl.convert_layout(scale.reshape((BM, BN // 64)), ep)
    rm = start_row + gl.arange(0, BM, gl.SliceLayout(1, ep))
    cn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, ep))
    sg = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, ep))
    gl.store(Y + rm[:, None] * (N // 4) + cn[None, :], packed)
    gl.store(YScale + rm[:, None] * (N // 64) + sg[None, :], scale)

@gluon.jit
def _reconstruction_bits(value, code, quantum):
    reconstructed = code.to(gl.float32) * quantum[:, None]
    mismatch = reconstructed.to(gl.uint32, bitcast=True) ^ value.to(gl.uint32, bitcast=True)
    return gl.max(mismatch, 1)

@gluon.jit
def _store_w2_panel(acc, raw_base, code_base, header_base, column, valid_rows, N: gl.constexpr, BN: gl.constexpr, ROW_BASE: gl.constexpr):
    peak = gl.max(gl.abs(acc), 1)
    exponent = (peak.to(gl.uint32, bitcast=True) >> 23 & 255).to(gl.int32) - 141
    exponent = gl.maximum(-126, gl.minimum(112, exponent))
    quantum = (exponent + 127 << 23).to(gl.float32, bitcast=True)
    inverse = (127 - exponent << 23).to(gl.float32, bitcast=True)
    biased = gl.fma(acc, inverse[:, None], 12582912.0)
    code = biased.to(gl.uint32, bitcast=True).to(gl.int16)
    if acc.shape[0] == 64 and ROW_BASE == 0:
        acc_low, acc_high = _split_columns(acc)
        code_low, code_high = _split_columns(code)
        half_quantum = gl.convert_layout(quantum, gl.SliceLayout(1, code_low.type.layout), assert_trivial=True)
        low_bits = _reconstruction_bits(acc_low, code_low, half_quantum)
        escaped = low_bits | _reconstruction_bits(acc_high, code_high, half_quantum) != 0
        escaped = gl.convert_layout(escaped, gl.SliceLayout(1, acc.type.layout), assert_trivial=True)
    else:
        escaped = _reconstruction_bits(acc, code, quantum) != 0
    native_rows = ROW_BASE + gl.arange(0, acc.shape[0], gl.SliceLayout(1, acc.type.layout))
    gl.store(header_base + native_rows * (N // BN) + column, gl.where(escaped, 0, exponent + 127).to(gl.uint8), native_rows < valid_rows)
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    codes_tile = gl.allocate_shared_memory(gl.int16, acc.shape, shared, code)
    codes = codes_tile.load(ep)
    rows = ROW_BASE + gl.arange(0, acc.shape[0], gl.SliceLayout(1, ep))
    columns = column * BN + gl.arange(0, BN, gl.SliceLayout(0, ep))
    gl.amd.cdna4.buffer_store(stored_value=codes, ptr=code_base, offsets=rows[:, None] * N + columns[None, :], mask=rows[:, None] < valid_rows, cache='.cs')
    gl.barrier()
    if gl.sum((escaped & (native_rows < valid_rows)).to(gl.int32), 0) != 0:
        raw_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
        raw_tile = gl.allocate_shared_memory(gl.float32, acc.shape, raw_layout, acc)
        raw = raw_tile.load(ep)
        escape_rows = gl.convert_layout(escaped, gl.SliceLayout(1, ep))
        gl.amd.cdna4.buffer_store(stored_value=raw, ptr=raw_base, offsets=rows[:, None] * N + columns[None, :], mask=(rows[:, None] < valid_rows) & escape_rows[:, None], cache='.cs')
    gl.barrier()

@gluon.jit
def _store_w2_panels(acc, raw_base, code_base, header_base, column, valid_rows, N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr):
    if BM == 128:
        low, high = gl.split(acc.reshape((2, 64, BN)).permute((1, 2, 0)))
        _store_w2_panel(low, raw_base, code_base, header_base, column, valid_rows, N, BN, 0)
        _store_w2_panel(high, raw_base, code_base, header_base, column, valid_rows, N, BN, 64)
    else:
        _store_w2_panel(acc, raw_base, code_base, header_base, column, valid_rows, N, BN, 0)

@gluon.jit(do_not_specialize=['M'])
def _fp4_gemm(X, XScale, W, WScale, Sorted, expert, start_row, column, valid_rows, N: gl.constexpr, K: gl.constexpr, UP: gl.constexpr, M, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[4, 1] if not UP and BM >= 64 else [2, 2])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0]) if UP else gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([16, 1], [1, 64], [2, 2], [0, 1])
    sl: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [BM, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    mi = gl.arange(0, BM, gl.SliceLayout(1, al))
    row = start_row + mi
    if UP:
        if expert == 256:
            row += M
        else:
            route = gl.load(Sorted + row)
            row = gl.maximum(route // 8, 0)
        row = gl.where(mi < valid_rows, row, 0)
    ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK // 2, gl.SliceLayout(1, bl))
    ni = gl.arange(0, BN, gl.SliceLayout(0, bl))
    if UP:
        wn = column * (BN // 2) + ni % (BN // 2) + ni // (BN // 2) * (N // 2)
    else:
        wn = column * BN + ni
    scale_row = gl.convert_layout(row, gl.SliceLayout(1, sl))
    scale_n = gl.convert_layout(wn, gl.SliceLayout(1, sl))
    sg = gl.arange(0, BK // 32, gl.SliceLayout(0, sl))
    sm = gl.arange(0, BM, gl.SliceLayout(1, sl))
    a_offset = row[:, None] * (K // 2) + ak[None, :]
    b_offset = wn[None, :] // 16 * (K * 8) + bk[:, None] // 16 * 256 + wn[None, :] % 16 * 16 + bk[:, None] % 16
    w_base = W + expert * (N * K // 2)
    ws_base = WScale + expert * (N * gl.cdiv(K // 32, 8) * 8)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    if UP:
        gl.static_assert(BK == 128 and BN == 256)
        a_shared: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, order=[1, 0])
        b_shared: gl.constexpr = gl.SharedLinearLayout(offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]], block_bases=[], alignment=16)
        a_tile = gl.allocate_shared_memory(gl.uint8, (BM, BK // 2), a_shared)
        b_tile = gl.allocate_shared_memory(gl.uint8, (BK // 2, BN), b_shared)
        xa = gl.load(XScale + scale_row[:, None] * (K // 32) + sg[None, :], sm[:, None] < valid_rows, 127)
        wb = gl.load(ws_base + _scale_offset(scale_n[:, None], sg[None, :], K))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(a_tile, X, a_offset)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(b_tile, w_base, b_offset)
        gl.amd.cdna4.async_copy.commit_group()
        for step in range(K // BK - 1):
            gl.amd.cdna4.async_copy.wait_group(0)
            a = a_tile.load(ad)
            b = b_tile.load(bd)
            a_scale = gl.convert_layout(xa, asl)
            b_scale = gl.convert_layout(wb, bsl)
            xa = gl.load(XScale + scale_row[:, None] * (K // 32) + (step + 1) * (BK // 32) + sg[None, :], sm[:, None] < valid_rows, 127)
            wb = gl.load(ws_base + _scale_offset(scale_n[:, None], (step + 1) * (BK // 32) + sg[None, :], K))
            gl.amd.cdna4.async_copy.buffer_load_to_shared(a_tile, X, a_offset + (step + 1) * (BK // 2))
            gl.amd.cdna4.async_copy.buffer_load_to_shared(b_tile, w_base, b_offset + (step + 1) * (BK * 8))
            gl.amd.cdna4.async_copy.commit_group()
            acc = gl.amd.cdna4.mfma_scaled(a, a_scale, 'e2m1', b, b_scale, 'e2m1', acc)
        gl.amd.cdna4.async_copy.wait_group(0)
        a = a_tile.load(ad)
        b = b_tile.load(bd)
        acc = gl.amd.cdna4.mfma_scaled(a, gl.convert_layout(xa, asl), 'e2m1', b, gl.convert_layout(wb, bsl), 'e2m1', acc)
    else:
        for step in range(K // BK):
            a = gl.load(X + a_offset + step * (BK // 2), mi[:, None] < valid_rows, 0)
            b = gl.load(w_base + b_offset + step * (BK * 8))
            xa = gl.load(XScale + scale_row[:, None] * (K // 32) + step * (BK // 32) + sg[None, :], sm[:, None] < valid_rows, 127)
            wb = gl.load(ws_base + _scale_offset(scale_n[:, None], step * (BK // 32) + sg[None, :], K))
            acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, ad), gl.convert_layout(xa, asl), 'e2m1', gl.convert_layout(b, bd), gl.convert_layout(wb, bsl), 'e2m1', acc)
    return acc

@gluon.jit(do_not_specialize=['M'])
def _expert_projection(X, XScale, W, Scales, Sorted, Experts, Y, YScale, Codes, Headers, N: gl.constexpr, K: gl.constexpr, UP: gl.constexpr, M, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUP_M: gl.constexpr):
    pid = gl.program_id(0)
    blocks = gl.cdiv((9 if UP else 8) * M + (257 if UP else 256) * (BM - 1), BM)
    block, column = _grouped_tile(pid, blocks, N // BN, GROUP_M)
    descriptor = gl.load(Experts + block)
    active = descriptor >= 0
    if not UP:
        active = active & (descriptor & 511 != 256)
    if active:
        expert = (descriptor & 511).to(gl.int32)
        valid_rows = (descriptor >> 9 & 255).to(gl.int32)
        dense_base = (descriptor >> 17).to(gl.int32)
        source_row = gl.where(UP & (expert == 256), dense_base - 8 * M, block * BM)
        acc = _fp4_gemm(X, XScale, W, Scales, Sorted, expert, source_row, column, valid_rows, N, K, UP, M, BM, BN, BK)
        if UP:
            _store_w13_activation(acc, Y, YScale, expert, block * BM, column, N, BM, BN)
        else:
            raw_base = Y + dense_base.to(gl.int64) * N
            code_base = Codes + dense_base.to(gl.int64) * N
            header_base = Headers + dense_base * (N // BN)
            _store_w2_panels(acc, raw_base, code_base, header_base, column, valid_rows, N, BM, BN)

@gluon.jit(do_not_specialize=['M'])
def _shared_reduce(X, XScale, W, Scales, Offsets, P, Codes, Headers, Weights, Inverse, Y, M, H: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, route_count=8, GROUP: gl.constexpr=1):
    blocks = gl.cdiv(M, BM)
    block, column = _grouped_tile(gl.program_id(0), blocks, H // BN, GROUP)
    first = block * BM
    source_row = gl.load(Offsets + 256) + first
    acc = _fp4_gemm(X, XScale, W, Scales, Offsets, 256, source_row, column, gl.minimum(M - first, BM), H, K, False, M, BM, BN, 256)
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    shared = gl.convert_layout(acc.to(gl.bfloat16), ep)
    rm = first + gl.arange(0, BM, gl.SliceLayout(1, ep))
    cn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, ep))
    value = gl.full((BM, BN), 0, gl.float32, ep)
    metadata_layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    metadata_row = first + gl.arange(0, BM, gl.SliceLayout(1, metadata_layout))
    route_id = gl.arange(0, 8, gl.SliceLayout(0, metadata_layout))
    dense_rows = gl.load(Inverse + metadata_row[:, None] * 8 + route_id[None, :], metadata_row[:, None] < M, 0)
    route_weights = gl.load(Weights + metadata_row[:, None] * 8 + route_id[None, :], metadata_row[:, None] < M, 0)
    route_headers = gl.load(Headers + dense_rows * (H // 256) + column // (256 // BN), metadata_row[:, None] < M, 127)
    for j in range(route_count):
        index = gl.full((BM, 1), j, gl.int32, metadata_layout)
        dense_row = gl.gather(dense_rows, index, 1).reshape((BM,))
        weight = gl.gather(route_weights, index, 1).reshape((BM,))
        dense_row = gl.convert_layout(dense_row, gl.SliceLayout(1, ep))
        weight = gl.convert_layout(weight, gl.SliceLayout(1, ep))
        header_bits = gl.gather(route_headers, index, 1).reshape((BM,))
        header_bits = gl.convert_layout(header_bits, gl.SliceLayout(1, ep))
        header = (header_bits.to(gl.uint32) << 23).to(gl.float32, bitcast=True)
        code_offset = dense_row[:, None] * H + cn[None, :]
        code = gl.load(Codes + code_offset, rm[:, None] < M, 0, cache_modifier='.cg')
        part = code.to(gl.float32) * header[:, None]
        part = gl.load(P + dense_row[:, None] * H + cn[None, :], (rm[:, None] < M) & (header[:, None] == 0), part, cache_modifier='.cg')
        value += part * weight[:, None]
    value += shared.to(gl.float32)
    gl.store(Y + rm[:, None] * H + cn[None, :], value, rm[:, None] < M)

class _Workspace:

    def __init__(self, x, intermediate, capacity, block_m, chunks):
        m, h = x.shape

        def empty(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=x.device)
        self.parts = empty((8 * m, h), torch.float32)
        self.codes = empty((8 * m, h), torch.int16)
        early_bf16 = self.parts.view(torch.bfloat16)
        early_i32 = self.parts.view(torch.int32)
        cursor = 0

        def early(shape, base, item_bytes):
            nonlocal cursor
            elements = 1
            for dimension in shape:
                elements *= dimension
            begin = triton.cdiv(cursor, 256) * 256
            cursor = begin + elements * item_bytes
            return base.view(-1).narrow(0, begin // item_bytes, elements).view(shape)
        self.logits = early((m, 256), early_bf16, 2)
        self.ids = early((m, 8), early_i32, 4)
        self.weights = empty((m, 8), torch.float32)
        self.partial_counts = early((256, chunks), early_i32, 4)
        self.counts = early((257,), early_i32, 4)
        self.offsets = empty((2 * 257,), torch.int32)
        self.inverse = empty((m, 8), torch.int32)
        self.sorted_routes = early((capacity,), early_i32, 4)
        self.experts = empty((capacity // block_m,), torch.int64)
        self.xq = self.codes.view(torch.uint8).as_strided((2 * m, h // 2), (h // 2, 1))
        self.xs = empty((2 * m, h // 32), torch.uint8)
        self.aq = empty((capacity, intermediate // 2), torch.uint8)
        self.aqs = empty((capacity, intermediate // 32), torch.uint8)
        self.headers = empty((8 * m, h // 256), torch.uint8)
        self.output = empty((m, h))

def _route_inputs(x, router, correction_bias, work, block_m):
    m, h = x.shape
    capacity = work.sorted_routes.numel()
    chunks = work.partial_counts.shape[1]
    routes = 8 * m
    router_rows = 128 if m >= 12288 else 64
    router_columns = 64
    router_k = 128
    quantize_groups, quantize_values = (128, 16) if m < 12288 else (256, 32)
    projection_tiles = triton.cdiv(m, router_rows) * (256 // router_columns)
    quantize_tiles = triton.cdiv(m * h // 32, quantize_groups)
    _router_project_quantize[projection_tiles + quantize_tiles,](x, router, work.logits, work.xq, work.xs, m, h, x.stride(0), router_rows, router_k, router_columns, quantize_groups, quantize_values)
    _router[m,](work.logits, correction_bias, work.ids, work.weights, num_warps=1)
    _chunk_counts[max(chunks, triton.cdiv(capacity, 1024)),](work.ids, work.partial_counts, work.sorted_routes, work.experts, routes, chunks, capacity, block_m)
    _chunk_prefix[257,](work.partial_counts, work.partial_counts, work.counts, chunks, triton.next_power_of_2(chunks), m)
    _build_expert_blocks[257,](work.counts, work.offsets, work.experts, block_m, triton.next_power_of_2(triton.cdiv(m, block_m)))
    _scatter[chunks,](work.ids, work.offsets, work.partial_counts, work.sorted_routes, work.inverse, routes, chunks)

def _padded_capacity(routes, experts, block_m):
    return triton.cdiv(routes + experts * (block_m - 1), block_m) * block_m

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale):
    m, h = x.shape
    assert 4193 <= m <= 16768
    intermediate = w13.shape[1] // 2
    block_m = 128
    capacity = _padded_capacity(9 * m, 257, block_m)
    work = _Workspace(x, intermediate, capacity, block_m, triton.cdiv(8 * m, 256))
    _route_inputs(x, router, correction_bias, work, block_m)
    up_columns, up_k = (256, 128)
    _expert_projection[capacity // block_m * (2 * intermediate // up_columns),](work.xq, work.xs, w13, w13_scale, work.sorted_routes, work.experts, work.aq, work.aqs, work.codes, work.headers, 2 * intermediate, h, True, m, block_m, up_columns, up_k, 1 if block_m == 64 else 8, enable_fp_fusion=False)
    routed_capacity = _padded_capacity(8 * m, 256, block_m)
    _expert_projection[routed_capacity // block_m * (h // 256),](work.aq, work.aqs, w2, w2_scale, work.sorted_routes, work.experts, work.parts, work.aqs, work.codes, work.headers, h, intermediate, False, m, block_m, 256, 256, 2, enable_fp_fusion=False, waves_per_eu=2)
    reduce_columns, reduce_rows = (256, 32)
    _shared_reduce[triton.cdiv(m, reduce_rows) * (h // reduce_columns),](work.aq, work.aqs, w2, w2_scale, work.offsets, work.parts, work.codes, work.headers, work.weights, work.inverse, work.output, m, h, intermediate, reduce_rows, reduce_columns, 8, 8 if m >= 12288 else 1, enable_fp_fusion=False)
    return work.output
