import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

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
def _quantize_group(x, shared):
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
    code = gl.where(a <= 0.25, 0, gl.where(a < 0.75, 1, gl.where(a <= 1.25, 2, gl.where(a < 1.75, 3, gl.where(a <= 2.5, 4, gl.where(a < 3.5, 5, gl.where(a <= 5.0, 6, 7)))))))
    code = (code | gl.where(x < 0, 8, 0)).to(gl.uint8)
    pack_layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [gl.num_warps(), 1], [1, 0])
    code = gl.convert_layout(code, pack_layout)
    lo, hi = gl.split(code.reshape((x.shape[0], 16, 2)))
    return (lo | hi << 4, (exponent + 127).to(gl.uint8))

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
        n = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, bl)).to(gl.uint32)
        bk = gl.arange(0, BK, gl.SliceLayout(0, bl))
        acc = gl.zeros((16, BN), gl.float32, mma)
        for base in range(H // SPLITS // BK):
            start = split * (H // SPLITS) + base * BK
            a = gl.load(X + input_m[:, None] * SX + start + ak[None, :])
            b = gl.load(W + n[:, None] * H + start + bk[None, :])
            acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b.T, gl.DotOperandLayout(1, mma, 8)), acc)
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma)).to(gl.uint32)
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
        routed, rs = _quantize_group(values, False)
        shared, ss = _quantize_group(values, True)
        qg = gl.convert_layout(group, gl.SliceLayout(1, routed.type.layout))
        qk = qg[:, None] * 16 + gl.arange(0, 16, gl.SliceLayout(0, routed.type.layout))[None, :]
        gl.store(Q + row * (H // 2) + qk, routed)
        gl.store(Q + (row + M) * (H // 2) + qk, shared)
        gl.store(QS + row * (H // 32) + group, rs)
        gl.store(QS + (row + M) * (H // 32) + group, ss)

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
def _expert_projection(X, XS, W, WS, Ids, Groups, Y, N: gl.constexpr, K: gl.constexpr, M: gl.constexpr, UP: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, SHARED_ONLY: gl.constexpr=False, STAGGER: gl.constexpr=False):
    if SHARED_ONLY:
        route = gl.cast(8, gl.uint32)
        expert = gl.cast(256, gl.uint32)
        if STAGGER:
            tile = gl.program_id(0).to(gl.uint32)
            split = gl.cast(0, gl.uint32)
        else:
            work = gl.program_id(0).to(gl.uint32) - M
            tile = work % (N // BN)
            split = work // (N // BN)
    else:
        if STAGGER:
            work = gl.program_id(0).to(gl.uint32) - M * (N // 128)
            tile = work % (N // BN)
            split = work // (N // BN) % SPLITS
            job = work // (N // BN * SPLITS)
            route = job // 8 * 9 + job % 8
        else:
            split = gl.program_id(0).to(gl.uint32)
            job = gl.program_id(1).to(gl.uint32)
            tile = gl.program_id(2).to(gl.uint32)
            route = gl.where(job < M * 8, job // 8 * 9 + job % 8, 8)
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
        dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
        dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
        al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [1, 0])
        bl: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [0, 1])
        sal: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [16, BK // 32])
        sbl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [BN, BK // 32])
        mi = gl.arange(0, 16, gl.SliceLayout(1, al))
        if M >= 2:
            rank = (members >> mi * 4 & 15).to(gl.int32) - 1
            destination = gl.where(rank >= 0, mi * 9 + rank, -1)
            source_route = gl.maximum(destination, route.to(gl.int32)).to(gl.uint32)
        else:
            destination = gl.full((16,), route, gl.int32, gl.SliceLayout(1, al))
            source_route = destination.to(gl.uint32)
        row = source_route // 9 + gl.where(expert == 256, M, 0) if UP else source_route
        sk_row = gl.convert_layout(row, gl.SliceLayout(1, sal))
        ak = gl.arange(0, BK // 2, gl.SliceLayout(0, al)).to(gl.uint32)
        n = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, bl)).to(gl.uint32)
        k = gl.arange(0, BK // 8, gl.SliceLayout(0, bl)).to(gl.uint32) * 8
        sn = tile * BN + gl.arange(0, BN, gl.SliceLayout(1, sbl)).to(gl.uint32)
        sag = gl.arange(0, BK // 32, gl.SliceLayout(0, sal)).to(gl.uint32)
        sbg = gl.arange(0, BK // 32, gl.SliceLayout(0, sbl)).to(gl.uint32)
        weight_base = W.to(gl.pointer_type(gl.uint32)) + expert.to(gl.int64) * (N * K // 8)
        scale_base = WS + expert * (N * gl.cdiv(K // 32, 8) * 8)
        acc = gl.zeros((16, BN), gl.float32, mma)
        for block in range(K // SPLITS // BK):
            start = split * (K // SPLITS) + block * BK
            a = gl.load(X + row[:, None] * (K // 2) + start // 2 + ak[None, :])
            offsets = _weight_offset(0, n[:, None], start + k[None, :], N, K) // 4
            offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
            words = gl.amd.cdna4.buffer_load(weight_base, offsets.to(gl.uint32))
            b0 = words.to(gl.uint8)
            b1 = (words >> 8).to(gl.uint8)
            b2 = (words >> 16).to(gl.uint8)
            b3 = (words >> 24).to(gl.uint8)
            b = gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((BN, BK // 2))
            xs = gl.load(XS + sk_row[:, None] * (K // 32) + start // 32 + sag[None, :])
            ws = gl.load(scale_base + _scale_offset(0, sn[:, None], start + sbg[None, :] * 32, N, K))
            acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, dot_a), xs, 'e2m1', gl.convert_layout(b.T, dot_b), ws, 'e2m1', acc)
        output_route = gl.convert_layout(destination, gl.SliceLayout(1, mma))
        om = gl.arange(0, 16, gl.SliceLayout(1, mma))
        on = tile * BN + gl.arange(0, BN, gl.SliceLayout(0, mma)).to(gl.uint32)
        valid = output_route >= 0 if M >= 2 else om == 0
        offsets = (output_route[:, None].to(gl.uint32) * SPLITS + split) * N + on[None, :]
        if SHARED_ONLY and STAGGER:
            offsets = output_route[:, None].to(gl.uint32) // 9 * N + on[None, :]
        gl.store(Y + offsets, acc, valid[:, None])

@gluon.jit
def _activation_tile(GU, Q, QS, route, tile, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [WARPS, 1], [1, 0])
    group = tile * (2 * WARPS) + gl.arange(0, 2 * WARPS, gl.SliceLayout(1, layout))
    lane = gl.arange(0, 32, gl.SliceLayout(0, layout))
    n = group[:, None] * 32 + lane[None, :]
    offsets = (route * SPLITS * 2 * I + n).to(gl.uint32)
    gate = gl.amd.cdna4.buffer_load(GU, offsets)
    up = gl.amd.cdna4.buffer_load(GU, offsets + I)
    for split in gl.static_range(1, SPLITS):
        gate += gl.amd.cdna4.buffer_load(GU, offsets + split * 2 * I)
        up += gl.amd.cdna4.buffer_load(GU, offsets + split * 2 * I + I)
    shared = route % 9 == 8
    gate = gl.where(shared, gate.to(gl.bfloat16).to(gl.float32), gate)
    up = gl.where(shared, up.to(gl.bfloat16).to(gl.float32), up)
    a = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16).to(gl.float32)
    q, s = _quantize_group(a, shared)
    qg = gl.convert_layout(group, gl.SliceLayout(1, q.type.layout))
    qk = qg[:, None] * 16 + gl.arange(0, 16, gl.SliceLayout(0, q.type.layout))[None, :]
    gl.store(Q + route * (I // 2) + qk, q)
    gl.store(QS + route * (I // 32) + group, s)

@gluon.jit
def _activate_quantize(GU, Q, QS, I: gl.constexpr, SPLITS: gl.constexpr, WARPS: gl.constexpr):
    _activation_tile(GU, Q, QS, gl.program_id(0), gl.program_id(1), I, SPLITS, WARPS)

@gluon.jit
def _select_and_shared(Logits, Bias, Ids, Weights, Groups, X, XS, W, WS, GU, M: gl.constexpr, H: gl.constexpr, I: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    if gl.program_id(0) < M:
        _select_routes(Logits, Bias, Ids, Weights, Groups, 12, M >= 2)
    else:
        _expert_projection(X, XS, W, WS, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, BK, SHARED_ONLY=True)

@gluon.jit
def _up_and_shared_activation(X, XS, W, WS, Ids, Groups, GU, AQ, AQS, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    program = gl.program_id(0)
    if program < M * (I // 64):
        route = program // (I // 64) * 9 + 8
        tile = program % (I // 64)
        _activation_tile(GU, AQ, AQS, route, tile, I, SPLITS, 1)
    else:
        _expert_projection(X, XS, W, WS, Ids, Groups, GU, 2 * I, H, M, True, SPLITS, BN, BK, STAGGER=True)

@gluon.jit
def _activation_and_shared_down(GU, AQ, AQS, W, WS, Ids, Groups, Shared, H: gl.constexpr, I: gl.constexpr, M: gl.constexpr, SPLITS: gl.constexpr):
    program = gl.program_id(0)
    if program < H // 64:
        _expert_projection(AQ, AQS, W, WS, Ids, Groups, Shared, H, I, M, False, 1, 64, 256, SHARED_ONLY=True, STAGGER=True)
    else:
        work = program - H // 64
        job = work // (I // 64)
        route = job // 8 * 9 + job % 8
        tile = work % (I // 64)
        _activation_tile(GU, AQ, AQS, route, tile, I, SPLITS, 1)

@gluon.jit
def _routed_down_batch(X, XS, W, WS, Ids, token, tile, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=False, warps_per_cta=[1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [1, 1], [1, 0])
    pl: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 1], [0, 1])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    sal: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_a, [16, K // 32])
    sbl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_b, [8 * BN, K // 32])
    ar = gl.minimum(gl.arange(0, 16, gl.SliceLayout(1, al)), 7).to(gl.uint32)
    ak = gl.arange(0, K // 2, gl.SliceLayout(0, al)).to(gl.uint32)
    pn = gl.arange(0, 8 * BN, gl.SliceLayout(1, pl)).to(gl.uint32)
    pk = gl.arange(0, K // 8, gl.SliceLayout(0, pl)).to(gl.uint32) * 8
    sr = gl.minimum(gl.arange(0, 16, gl.SliceLayout(1, sal)), 7).to(gl.uint32)
    sag = gl.arange(0, K // 32, gl.SliceLayout(0, sal)).to(gl.uint32)
    sn = gl.arange(0, 8 * BN, gl.SliceLayout(1, sbl)).to(gl.uint32)
    sbg = gl.arange(0, K // 32, gl.SliceLayout(0, sbl)).to(gl.uint32)
    expert = gl.load(Ids + token * 9 + pn // BN).to(gl.uint32)
    se = gl.load(Ids + token * 9 + sn // BN).to(gl.uint32)
    a = gl.load(X + (token * 9 + ar[:, None]) * (K // 2) + ak[None, :])
    offsets = _weight_offset(expert[:, None], tile * BN + pn[:, None] % BN, pk[None, :], N, K) // 4
    offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offsets)
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    b = gl.join(gl.join(b0, b2), gl.join(b1, b3)).reshape((8 * BN, K // 2))
    xs = gl.load(XS + (token * 9 + sr[:, None]) * (K // 32) + sag[None, :])
    ws = gl.load(WS + _scale_offset(se[:, None], tile * BN + sn[:, None] % BN, sbg[None, :] * 32, N, K))
    acc = gl.zeros((16, 8 * BN), gl.float32, mma)
    acc = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, dot_a), xs, 'e2m1', gl.convert_layout(b.T, dot_b), ws, 'e2m1', acc)
    n = gl.arange(0, 8 * BN, gl.SliceLayout(0, mma))
    matching_route = (n // BN)[None, :]
    return gl.gather(acc, matching_route, 0).reshape((8, BN))

@gluon.jit
def _down_combine(X, XS, W, Scales, Ids, Weights, Shared, Y, H: gl.constexpr, I: gl.constexpr, BN: gl.constexpr):
    tile = gl.program_id(0).to(gl.uint32)
    token = gl.program_id(1).to(gl.uint32)
    parts = _routed_down_batch(X, XS, W, Scales, Ids, token, tile, H, I, BN)
    output_regs: gl.constexpr = [[1, 0], [2, 0], [4, 0]] + ([[0, 16]] if BN == 32 else [])
    layout: gl.constexpr = gl.DistributedLinearLayout(reg_bases=output_regs, lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 0], [0, 0]], warp_bases=[], block_bases=[], shape=[8, BN])
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
    stagger = m <= 16
    splits = 12 if stagger else 6
    up_width = (32 if m <= 2 or m >= 7 else 64) if stagger else 16
    up_block = (256 if m >= 7 else 512) if stagger else 128
    down_width = 16 if m <= 2 else 32

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=x.device)
    logits = empty((m, 12, 256), torch.float32)
    ids = empty((routes,), torch.int32)
    groups = empty((257,), torch.uint64)
    weights = empty((routes,), torch.float32)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    gu = empty((routes, splits, 2 * intermediate), torch.float32)
    aq = empty((routes, intermediate // 2), torch.uint8)
    aqs = empty((routes, intermediate // 32), torch.uint8)
    parts = empty((m if stagger else routes, h), torch.float32)
    out = empty((m, h))
    _router_linear[192 + m * (h // 128) + int(m >= 2),](x, router, logits, xq, xs, groups, h, x.stride(0), m, 12, 512, 16, m >= 2, 2, 4, num_warps=1)
    if stagger:
        _select_and_shared[m + splits * (2 * intermediate // up_width),](logits, correction_bias, ids, weights, groups, xq, xs, w13, w13_scale, gu, m, h, intermediate, splits, up_width, up_block, num_warps=1)
        up_jobs = m * (intermediate // 64) + m * 8 * splits * (2 * intermediate // up_width)
        _up_and_shared_activation[up_jobs,](xq, xs, w13, w13_scale, ids, groups, gu, aq, aqs, h, intermediate, m, splits, up_width, up_block, num_warps=1, enable_fp_fusion=False)
        _activation_and_shared_down[h // 64 + m * 8 * (intermediate // 64),](gu, aq, aqs, w2, w2_scale, ids, groups, parts, h, intermediate, m, splits, num_warps=1, enable_fp_fusion=False)
        _down_combine[h // down_width, m](aq, aqs, w2, w2_scale, ids, weights, parts, out, h, intermediate, down_width, num_warps=1, enable_fp_fusion=False)
    else:
        _select_routes[m,](logits, correction_bias, ids, weights, groups, 12, True, num_warps=1)
        _expert_projection[splits, m * 8 + 1, 2 * intermediate // up_width](xq, xs, w13, w13_scale, ids, groups, gu, 2 * intermediate, h, m, True, splits, up_width, up_block, num_warps=1)
        _activate_quantize[routes, intermediate // 64](gu, aq, aqs, intermediate, splits, 1, num_warps=1, enable_fp_fusion=False)
        _expert_projection[1, m * 8 + 1, h // 64](aq, aqs, w2, w2_scale, ids, groups, parts, h, intermediate, m, False, 1, 64, 256, num_warps=1)
        _combine[m, h // 256](parts, weights, out, h, 256, 1, 4, num_warps=4, enable_fp_fusion=False)
    return out
