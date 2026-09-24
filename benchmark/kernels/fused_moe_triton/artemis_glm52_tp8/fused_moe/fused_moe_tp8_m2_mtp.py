import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _pack_fp4(x, a, exponent):
    code = gl.where(a <= 0.25, 0, gl.where(a < 0.75, 1, gl.where(a <= 1.25, 2, gl.where(a < 1.75, 3, gl.where(a <= 2.5, 4, gl.where(a < 3.5, 5, gl.where(a <= 5.0, 6, 7)))))))
    code = (code | gl.where(x < 0, 8, 0)).to(gl.uint8)
    pack_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [gl.num_warps(), 1], [1, 0])
    code = gl.convert_layout(code, pack_layout)
    lo, hi = gl.split(code.reshape((x.shape[0], 16, 2)))
    return (lo | hi << 4, (exponent + 127).to(gl.uint8))

@gluon.jit
def _weight_offset(expert, n, k, N: gl.constexpr, K: gl.constexpr):
    byte = k // 2
    return (((expert * (N // 16) + n // 16) * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 256 + n % 16 * 16 + byte % 16

@gluon.jit
def _scale_offset(expert, n, k, N: gl.constexpr, K: gl.constexpr):
    group = k // 32
    groups: gl.constexpr = triton.cdiv(K // 32, 8) * 8
    return (((((expert * (N // 32) + n // 32) * (groups // 8) + group // 8) * 4 + group % 4) * 16 + n % 16) * 2 + group // 4 % 2) * 2 + n // 16 % 2

@gluon.jit
def _round_group(x, shared):
    peak = gl.max(gl.abs(x), 1)
    divided = gl.div_rn(peak, 6.0)
    bits = divided.to(gl.uint32, bitcast=True)
    exponent = (bits >> 23 & 255).to(gl.int32) - 127 + (bits & 8388607 != 0)
    peak_bits = peak.to(gl.uint32, bitcast=True)
    floor_exponent = (peak_bits >> 23 & 255).to(gl.int32) - 127
    threshold = gl.exp2(floor_exponent.to(gl.float32)) * 1.75
    even_exponent = floor_exponent - 2 + (peak >= threshold).to(gl.int32)
    exponent = gl.where(shared, even_exponent, exponent)
    exponent = gl.maximum(-127, gl.minimum(127, exponent))
    scale = gl.exp2(exponent.to(gl.float32))
    a = gl.abs(x / scale[:, None])
    return _pack_fp4(x, a, exponent)

@gluon.jit
def _router_linear(X, W, Y, Q, QS, Groups, H: gl.constexpr, SX: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr, GROUPED: gl.constexpr, QVEC: gl.constexpr, QGROUPS: gl.constexpr):
    program = gl.program_id(0)
    if program < 256 // BN * SPLITS:
        tile = program % (256 // BN)
        split = program // (256 // BN)
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
        al: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 1], [0, 1])
        bl: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
        mi = gl.arange(0, 16, gl.SliceLayout(1, al))
        input_m = gl.minimum(mi, M - 1)
        ak = gl.arange(0, BK, gl.SliceLayout(0, al))
        n = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, bl))
        bk = gl.arange(0, BK, gl.SliceLayout(0, bl))
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in range(H // SPLITS // BK):
            start = split * (H // SPLITS) + base * BK
            a = gl.load(X + input_m[:, None] * SX + start + ak[None, :])
            b = gl.load(W + n[:, None] * H + start + bk[None, :])
            acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b.T, gl.DotOperandLayout(1, mma, 8)), acc)
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
        gl.store(Y + (om[:, None] * SPLITS + split) * 256 + on[None, :], acc, om[:, None] < M)
    elif GROUPED and program == 256 // BN * SPLITS:
        e = gl.arange(0, 512, gl.BlockedLayout([1], [64], [1], [0]))
        gl.store(Groups + e, 0, e < 257)
    else:
        quant = program - 256 // BN * SPLITS - (1 if GROUPED else 0)
        row = quant // (H // (32 * QGROUPS))
        tile_q = quant % (H // (32 * QGROUPS))
        layout: gl.constexpr = gl.BlockedLayout([1, QVEC], [2 * QVEC, 32 // QVEC], [1, 1], [1, 0])
        group = tile_q * QGROUPS + gl.arange(0, QGROUPS, gl.SliceLayout(1, layout))
        lane = gl.arange(0, 32, gl.SliceLayout(0, layout))
        k = group[:, None] * 32 + lane[None, :]
        values = gl.load(X + row * SX + k).to(gl.float32)
        routed, routed_scale = _round_group(values, False)
        shared, shared_scale = _round_group(values, True)
        qlayout: gl.constexpr = routed.type.layout
        qgroup = tile_q * QGROUPS + gl.arange(0, QGROUPS, gl.SliceLayout(1, qlayout))
        qlane = gl.arange(0, 16, gl.SliceLayout(0, qlayout))
        qoffset = qgroup[:, None] * 16 + qlane[None, :]
        gl.store(Q + row * (H // 2) + qoffset, routed)
        gl.store(Q + (row + M) * (H // 2) + qoffset, shared)
        gl.store(QS + row * (H // 32) + group, routed_scale)
        gl.store(QS + (row + M) * (H // 32) + group, shared_scale)

@gluon.jit
def _select_routes(Logits, Bias, Ids, Weights, Groups, SPLITS: gl.constexpr, GROUPED: gl.constexpr):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
    record_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    e = gl.arange(0, 256, layout)
    logits = gl.load(Logits + row * SPLITS * 256 + e)
    for part in gl.static_range(1, SPLITS):
        logits += gl.load(Logits + (row * SPLITS + part) * 256 + e)
    logits = logits.to(gl.bfloat16).to(gl.float32)
    probability = 1.0 / (1.0 + gl.exp(-logits))
    score = probability + gl.load(Bias + e).to(gl.float32)
    probability_table = gl.allocate_shared_memory(gl.float32, [256], gl.SwizzledSharedLayout(1, 1, 1, [0]), probability)
    record_size: gl.constexpr = 16 if GROUPED else 256
    r = gl.arange(0, record_size, record_layout)
    available = gl.full((256,), True, gl.int1, layout)
    total = 0.0
    selected_ids = gl.full((record_size,), 256, gl.int32, record_layout)
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        key = gl.where(available, e + gl.where(score == maximum, 0, 256), 512)
        local_key = gl.min(gl.reshape(key, (64, 4)), 1)
        local_key = gl.convert_layout(local_key, record_layout)
        valid, fallback = gl.inline_asm_elementwise('v_cmp_gt_u32_e64 $0, 1, $2\nv_cmp_gt_u32_e64 $1, 2, $2', constraints='=&s,=&s,v', args=[local_key >> 8], dtype=(gl.uint64, gl.uint64), is_pure=True, pack=1)
        mask = gl.where(valid != 0, valid, fallback)
        winning_lane = gl.inline_asm_elementwise('s_ff1_i32_b64 $0, $1', constraints='=s,s', args=[mask], dtype=gl.int32, is_pure=True, pack=1)
        elected = gl.inline_asm_elementwise('v_readlane_b32 $0, $1, $2', constraints='=s,v,s', args=[local_key, winning_lane], dtype=gl.int32, is_pure=True, pack=1)
        first = gl.full((1,), 0, gl.int32, record_layout)
        idx = gl.sum(gl.gather(elected, first, 0), 0) & 255
        selected_ids = gl.where(r == j, idx, selected_ids)
        available = available & (e != idx)
        score = gl.where(e == idx, -float('inf'), score)
    selected = probability_table.gather(gl.minimum(selected_ids, 255), 0)
    for j in gl.static_range(8):
        index = gl.full((1,), j, gl.int32, record_layout)
        total += gl.sum(gl.gather(selected, index, 0), 0)
    gl.store(Ids + row * 9 + r, selected_ids, r < 9)
    if GROUPED:
        membership = (r + 1).to(gl.uint64) << row * 4
        gl.amd.cdna4.buffer_atomic_or(Groups.to(gl.pointer_type(gl.int64)), selected_ids, membership.to(gl.int64, bitcast=True), r < 9, sem='relaxed')
    gl.store(Weights + row * 9 + r, selected / total * 2.5, r < 8)
    gl.store(Weights + row * 9 + 8, 1.0)

@gluon.jit
def _expert_coordinates(N: gl.constexpr, M: gl.constexpr, UP: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SHARED_ONLY: gl.constexpr, STAGGER: gl.constexpr):
    if SHARED_ONLY:
        route = gl.cast(8, gl.int32)
        if STAGGER:
            tile = gl.program_id(0)
            split = gl.cast(0, gl.int32)
        else:
            work = gl.program_id(0) - M
            tile = work % (N // BN)
            split = work // (N // BN)
    else:
        if STAGGER:
            work = gl.program_id(0) - M * (N // 128)
            job = work // (N // BN) % (M * 8) if M <= 2 else work // (SPLITS * (N // BN))
            tile = work % (N // BN) if M <= 2 else work // SPLITS % (N // BN)
            split = work // (M * 8 * (N // BN)) if M <= 2 else work % SPLITS
        else:
            job = gl.program_id(1) if UP else gl.program_id(0)
            tile = gl.program_id(2) if UP else gl.program_id(1)
            split = gl.program_id(0) if UP else gl.program_id(2)
        if STAGGER:
            route = job // 8 * 9 + job % 8
        else:
            route = gl.where(job < M * 8, job // 8 * 9 + job % 8, 8)
        if UP and M == 2 and (SPLITS % 4 == 0):
            linear = tile + N // BN * split
            tile = linear // 4 % (N // BN)
            split = linear % 4 + linear // (4 * (N // BN)) * 4
    return (route, tile, split)

@gluon.jit
def _scale_code(words, shift):
    return (words >> shift & 255).to(gl.uint8)

@gluon.jit
def _scaled_projection(X, XScales, W, Scales, Ids, Groups, Y, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, SHARED_ONLY: gl.constexpr=False, STAGGER: gl.constexpr=False, UP: gl.constexpr=True):
    BK: gl.constexpr = 128
    route, tile, split = _expert_coordinates(N, M, UP, SPLITS, BN, SHARED_ONLY, STAGGER)
    if SHARED_ONLY:
        expert = gl.cast(256, gl.uint32)
    else:
        expert = gl.load(Ids + route).to(gl.uint32)
    if M >= 2:
        if SHARED_ONLY:
            members = gl.cast(((1 << M * 4) - 1) // 15 * 9, gl.uint64)
        else:
            members = gl.load(Groups + expert)
        earlier = gl.cast(1, gl.uint64) << route // 9 * 4
        owner = members & earlier - 1 == 0
    else:
        owner = True
    if owner:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 1])
        load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [0, 1])
        word_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [0, 1])
        scale_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, 1], [0, 1])
        a_dot: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
        b_dot: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
        a_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(a_dot, [16, 4])
        b_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(b_dot, [BN, 4])
        mi = gl.arange(0, 16, gl.SliceLayout(1, load_layout))
        if M >= 2:
            rank = (members >> mi * 4 & 15).to(gl.int32) - 1
            destination = gl.where(rank >= 0, mi * 9 + rank, -1)
            source_route = gl.maximum(destination, route).to(gl.uint32)
        else:
            destination = gl.full((16,), route, gl.int32, gl.SliceLayout(1, load_layout))
            source_route = destination.to(gl.uint32)
        row = source_route // 9 + gl.where(expert == 256, M, 0) if UP else source_route
        byte_k = gl.arange(0, BK // 2, gl.SliceLayout(0, load_layout)).to(gl.uint32)
        wn = (tile * BN + gl.arange(0, BN, gl.SliceLayout(1, word_layout))).to(gl.uint32)
        wk = gl.arange(0, BK // 8, gl.SliceLayout(0, word_layout)).to(gl.uint32) * 8
        sn = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, scale_layout))
        sg = gl.arange(0, 4, gl.SliceLayout(0, scale_layout))
        sr = gl.convert_layout(row, gl.SliceLayout(1, scale_layout))
        weight_base = W.to(gl.pointer_type(gl.uint32)) + expert.to(gl.int64) * (N * K // 8)
        scale_base = Scales + expert * (N * gl.cdiv(K // 32, 8) * 8)
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in gl.static_range(K // SPLITS // BK):
            start = (split * (K // SPLITS) + base * BK).to(gl.uint32)
            a = gl.load(X + row[:, None] * (K // 2) + start // 2 + byte_k[None, :])
            offsets = _weight_offset(0, wn[:, None], start + wk[None, :], N, K) // 4
            offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
            words = gl.amd.cdna4.buffer_load(weight_base, offsets)
            b0 = words.to(gl.uint8)
            b1 = (words >> 8).to(gl.uint8)
            b2 = (words >> 16).to(gl.uint8)
            b3 = (words >> 24).to(gl.uint8)
            packed = gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((BN, BK // 2))
            b = gl.convert_layout(packed, load_layout)
            a_scale = gl.load(XScales + sr[:, None] * (K // 32) + start // 32 + sg[None, :])
            if base % 2 == 0:
                scale_offsets = _scale_offset(0, sn[:, None], start + 32 * sg[None, :], N, K)
                scale_words = gl.amd.cdna4.buffer_load(scale_base.to(gl.pointer_type(gl.uint32)), (scale_offsets // 4).to(gl.uint32))
            shift = (sn[:, None] // 16 % 2 + start // 128 % 2 * 2) * 8
            b_scale = _scale_code(scale_words, shift)
            acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, a_dot), gl.convert_layout(a_scale, a_scale_layout), 'e2m1', gl.convert_layout(b.T, b_dot), gl.convert_layout(b_scale, b_scale_layout), 'e2m1', acc)
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
        output_route = gl.convert_layout(destination, gl.SliceLayout(1, mma))
        valid = output_route >= 0 if M >= 2 else om == 0
        offsets = (output_route[:, None] * SPLITS + split) * N + on[None, :]
        if not UP and STAGGER and SHARED_ONLY:
            offsets = output_route[:, None] // 9 * N + on[None, :]
        gl.store(Y + offsets, acc, valid[:, None])

@gluon.jit
def _select_and_shared(Logits, Bias, Ids, Weights, Groups, X, XS, W, Scales, GU, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, ROUTER_SPLITS: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr):
    if gl.program_id(0) < M:
        _select_routes(Logits, Bias, Ids, Weights, Groups, ROUTER_SPLITS, M >= 2)
    else:
        _scaled_projection(X, XS, W, Scales, Ids, Groups, GU, 2 * I, H, M, SPLITS, BN, SHARED_ONLY=True)

@gluon.jit
def _activation_tile(GU, Q, QS, route, tile, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [WARPS, 1], [1, 0])
    group = tile * (2 * WARPS) + gl.arange(0, 2 * WARPS, gl.SliceLayout(1, layout))
    lane = gl.arange(0, 32, gl.SliceLayout(0, layout))
    n = group[:, None] * 32 + lane[None, :]
    gate = gl.load(GU + route * SPLITS * 2 * I + n)
    up = gl.load(GU + route * SPLITS * 2 * I + I + n)
    for split in gl.static_range(1, SPLITS):
        gate += gl.load(GU + (route * SPLITS + split) * 2 * I + n)
        up += gl.load(GU + (route * SPLITS + split) * 2 * I + I + n)
    shared = route % 9 == 8
    gate = gl.where(shared, gate.to(gl.bfloat16).to(gl.float32), gate)
    up = gl.where(shared, up.to(gl.bfloat16).to(gl.float32), up)
    a = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
    q, scale = _round_group(a, shared)
    qlayout: gl.constexpr = q.type.layout
    qgroup = tile * (2 * WARPS) + gl.arange(0, 2 * WARPS, gl.SliceLayout(1, qlayout))
    qlane = gl.arange(0, 16, gl.SliceLayout(0, qlayout))
    gl.store(Q + route * (I // 2) + qgroup[:, None] * 16 + qlane[None, :], q)
    gl.store(QS + route * (I // 32) + group, scale)

@gluon.jit
def _activate_quantize(GU, Q, QS, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr):
    _activation_tile(GU, Q, QS, gl.program_id(0), gl.program_id(1), I, SPLITS, WARPS)

@gluon.jit
def _up_and_shared_activation(X, XS, W, Scales, Ids, Groups, GU, AQ, AQS, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr):
    program = gl.program_id(0)
    if program < M * (I // 64):
        route = program // (I // 64) * 9 + 8
        tile = program % (I // 64)
        _activation_tile(GU, AQ, AQS, route, tile, I, SPLITS, 1)
    else:
        _scaled_projection(X, XS, W, Scales, Ids, Groups, GU, 2 * I, H, M, SPLITS, BN, STAGGER=True)

@gluon.jit
def _activation_and_shared_down(GU, AQ, AQS, W, Scales, Ids, Groups, Shared, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr):
    BN: gl.constexpr = 32 if M <= 2 else 16
    program = gl.program_id(0)
    if program < H // BN:
        _scaled_projection(AQ, AQS, W, Scales, Ids, Groups, Shared, H, I, M, 1, BN, SHARED_ONLY=True, STAGGER=True, UP=False)
    else:
        work = program - H // BN
        job = work // (I // 64)
        route = job // 8 * 9 + job % 8
        tile = work % (I // 64)
        _activation_tile(GU, AQ, AQS, route, tile, I, SPLITS, 1)

@gluon.jit
def _routed_down_batch(X, XS, W, Scales, Ids, token, tile, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr):
    BK: gl.constexpr = 128
    WIDTH: gl.constexpr = 8 * BN
    WARPS: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, WARPS], tiles_per_warp=[1, BN // 16])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, WARPS], [0, 1])
    extra_rows: gl.constexpr = [[16, 0], [32, 0]] if BN == 64 else [[16, 0]] if BN == 32 else []
    extra_routes: gl.constexpr = [[2 * BN, 0], [4 * BN, 0]] if WARPS == 2 else [[4 * BN, 0]] if WARPS == 4 else []
    wave_rows: gl.constexpr = [[BN, 0]] if WARPS == 2 else [[BN, 0], [2 * BN, 0]] if WARPS == 4 else [[BN, 0], [2 * BN, 0], [4 * BN, 0]]
    pl: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2]] + extra_rows + extra_routes, lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp_bases=wave_rows, block_bases=[], shape=[WIDTH, BK // 8])
    a_dot: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    b_dot: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    a_sl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(a_dot, [16, BK // 32])
    b_sl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(b_dot, [WIDTH, BK // 32])
    am = gl.arange(0, 16, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al))
    col = gl.arange(0, WIDTH, gl.SliceLayout(1, pl))
    pn = tile * BN + col % BN
    pk = gl.arange(0, BK // 8, gl.SliceLayout(0, pl)) * 8
    expert = gl.load(Ids + token * 9 + col // BN)
    sm = gl.arange(0, 16, gl.SliceLayout(1, a_sl))
    ask = gl.arange(0, BK // 32, gl.SliceLayout(0, a_sl))
    sc = gl.arange(0, WIDTH, gl.SliceLayout(1, b_sl))
    sn = tile * BN + sc % BN
    bsk = gl.arange(0, BK // 32, gl.SliceLayout(0, b_sl))
    se = gl.convert_layout(expert, gl.SliceLayout(1, b_sl))
    acc = gl.zeros((16, WIDTH), gl.float32, mma)
    for base in gl.static_range(K // BK):
        start = base * BK
        a = gl.load(X + (token * 9 + am[:, None] % 8) * (K // 2) + start // 2 + ak[None, :])
        offsets = _weight_offset(expert[:, None], pn[:, None], start + pk[None, :], N, K) // 4
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
        words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offsets.to(gl.uint32))
        b0 = words.to(gl.uint8)
        b1 = (words >> 8).to(gl.uint8)
        b2 = (words >> 16).to(gl.uint8)
        b3 = (words >> 24).to(gl.uint8)
        b = gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((WIDTH, BK // 2))
        a_scale = gl.load(XS + (token * 9 + sm[:, None] % 8) * (K // 32) + start // 32 + ask[None, :])
        if base % 2 == 0:
            scale_offsets = _scale_offset(se[:, None], sn[:, None], start + bsk[None, :] * 32, N, K)
            scale_words = gl.amd.cdna4.buffer_load(Scales.to(gl.pointer_type(gl.uint32)), (scale_offsets // 4).to(gl.uint32))
        shift = (sn[:, None] // 16 % 2 + base % 2 * 2) * 8
        b_scale = _scale_code(scale_words, shift)
        acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, a_dot), a_scale, 'e2m1', gl.convert_layout(b.T, b_dot), b_scale, 'e2m1', acc)
    out_col = gl.arange(0, WIDTH, gl.SliceLayout(0, mma))
    diagonal = gl.gather(acc, (out_col // BN)[None, :], 0)
    return gl.reshape(diagonal, (8, BN))

@gluon.jit
def _down_combine(X, XS, W, Scales, Ids, Weights, Shared, Y, H: gl.constexpr, I: gl.constexpr, BN: gl.constexpr):
    tile = gl.program_id(0)
    token = gl.program_id(1)
    parts = _routed_down_batch(X, XS, W, Scales, Ids, token, tile, H, I, BN)
    layout: gl.constexpr = gl.DistributedLinearLayout(reg_bases=[[1, 0], [2, 0], [4, 0]] + ([[0, 16], [0, 32]] if BN == 64 else [[0, 16]] if BN == 32 else []), lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 0], [0, 0]], warp_bases=[[0, 0]] * (3 if gl.num_warps() == 8 else 2 if gl.num_warps() == 4 else 1), block_bases=[], shape=[8, BN])
    parts = gl.convert_layout(parts, layout)
    value = gl.full((BN,), 0.0, gl.float32, gl.SliceLayout(0, layout))
    for rank in gl.static_range(8):
        part = gl.sum(gl.amd.slice(parts, [1, BN], [rank, 0]), 0)
        value += part * gl.load(Weights + token * 9 + rank)
    n = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, layout))
    shared_value = gl.load(Shared + token * H + n)
    shared_value = shared_value.to(gl.bfloat16).to(gl.float32)
    value += shared_value * gl.load(Weights + token * 9 + 8)
    gl.store(Y + token * H + n, value)

@gluon.jit
def _combine(P, Weights, Y, H: gl.constexpr, BLOCK: gl.constexpr, VEC: gl.constexpr, WARPS: gl.constexpr):
    row = gl.program_id(0)
    h = gl.program_id(1) * BLOCK + gl.arange(0, BLOCK, gl.BlockedLayout([VEC], [64], [WARPS], [0]))
    value = gl.full((BLOCK,), 0.0, gl.float32, gl.BlockedLayout([VEC], [64], [WARPS], [0]))
    for j in gl.static_range(9):
        part = gl.load(P + (row * 9 + j) * H + h)
        if j == 8:
            part = part.to(gl.bfloat16).to(gl.float32)
        value += part * gl.load(Weights + row * 9 + j)
    gl.store(Y + row * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale) -> torch.Tensor:
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    routes = m * 9
    grouped = m >= 2
    stagger = m <= 8
    splits = 12
    router_splits = 12
    router_block = 512
    quant_vector = 2 if 3 <= m <= 8 else 1
    quant_groups = 2 * quant_vector
    up_width = 32
    down_width = 32

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=x.device)
    logits = empty((m, router_splits, 256), torch.float32)
    ids = empty((routes,), torch.int32)
    groups = empty((257,), torch.uint64)
    weights = empty((routes,), torch.float32)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    gu = empty((routes, splits, 2 * intermediate), torch.float32)
    aq = empty((routes, intermediate // 2), torch.uint8)
    aqs = empty((routes, intermediate // 32), torch.uint8)
    down_output = empty((m if stagger else routes, h), torch.float32)
    out = empty((m, h))
    _router_linear[16 * router_splits + m * (h // (32 * quant_groups)) + int(grouped),](x, router, logits, xq, xs, groups, h, x.stride(0), m, router_splits, router_block, 16, grouped, quant_vector, quant_groups, num_warps=1)
    if stagger:
        _select_and_shared[m + splits * (2 * intermediate // up_width),](logits, correction_bias, ids, weights, groups, xq, xs, w13, w13_scale, gu, m, h, intermediate, router_splits, splits, up_width, num_warps=1)
    else:
        _select_routes[m,](logits, correction_bias, ids, weights, groups, router_splits, grouped, num_warps=1)
    if stagger:
        up_jobs = m * (intermediate // 64) + m * 8 * splits * (2 * intermediate // up_width)
        _up_and_shared_activation[up_jobs,](xq, xs, w13, w13_scale, ids, groups, gu, aq, aqs, h, intermediate, m, splits, up_width, num_warps=1, enable_fp_fusion=False)
        shared_width = 32 if m <= 2 else 16
        _activation_and_shared_down[h // shared_width + m * 8 * (intermediate // 64),](gu, aq, aqs, w2, w2_scale, ids, groups, down_output, h, intermediate, m, splits, num_warps=1, enable_fp_fusion=False)
        tail_width = 32
        _down_combine[h // tail_width, m](aq, aqs, w2, w2_scale, ids, weights, down_output, out, h, intermediate, tail_width, num_warps=4, enable_fp_fusion=False)
    else:
        jobs = m * 8 + 1
        activation_warps = 4
        up_grid = (splits, jobs, 2 * intermediate // up_width)
        _scaled_projection[up_grid](xq, xs, w13, w13_scale, ids, groups, gu, 2 * intermediate, h, m, splits, up_width, num_warps=1)
        _activate_quantize[routes, intermediate // (64 * activation_warps)](gu, aq, aqs, intermediate, splits, activation_warps, num_warps=activation_warps, enable_fp_fusion=False)
        down_grid = (jobs, h // down_width, 1)
        _scaled_projection[down_grid](aq, aqs, w2, w2_scale, ids, groups, down_output, h, intermediate, m, 1, down_width, UP=False, num_warps=1)
        _combine[m, h // 256](down_output, weights, out, h, 256, 1, 4, num_warps=4, enable_fp_fusion=False)
    return out
