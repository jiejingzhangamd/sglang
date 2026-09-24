"""Shared TP4 fused-MoE specialization for active batches M=128 and M=256."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)

@gluon.jit
def _add(a, b):
    return a + b

@gluon.jit
def _encode_groups(x, shared):
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
    inverse_scale = gl.div_rn(1.0, scale)
    a = gl.abs(x * inverse_scale[:, None])
    low, high = gl.split(a.reshape((x.shape[0], 16, 2)))
    packed = gl.inline_asm_elementwise('v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3;', constraints='=v,v,v,v', args=[low, high, gl.full(low.shape, 1.0, gl.float32, low.type.layout)], dtype=gl.uint32, is_pure=True, pack=1)
    signs = gl.where(x < 0, 8, 0).to(gl.uint8)
    low_sign, high_sign = gl.split(signs.reshape((x.shape[0], 16, 2)))
    return (packed.to(gl.uint8) & 119 | low_sign | high_sign << 4, (exponent + 127).to(gl.uint8))

@gluon.jit
def _quantize_input(X, Q, QS, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, GROUPS: gl.constexpr, CTA_OFFSET: gl.constexpr=0, WARPS: gl.constexpr=1):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [WARPS, 1], [1, 0])
    g = (gl.program_id(0) - CTA_OFFSET) * GROUPS + gl.arange(0, GROUPS, gl.SliceLayout(1, layout))
    k = gl.arange(0, 32, gl.SliceLayout(0, layout))
    x = gl.load(X + (g // (H // 32))[:, None] * SX + (g % (H // 32))[:, None] * 32 + k[None, :], g[:, None] < M * (H // 32), 0).to(gl.float32)
    rq, rs = _encode_groups(x, False)
    sq, ss = _encode_groups(x, True)
    pk = gl.arange(0, 16, gl.SliceLayout(0, rq.type.layout))
    pg = gl.convert_layout(g, gl.SliceLayout(1, rq.type.layout))
    gl.store(Q + pg[:, None] * 16 + pk[None, :], rq, pg[:, None] < M * (H // 32))
    gl.store(Q + M * H // 2 + pg[:, None] * 16 + pk[None, :], sq, pg[:, None] < M * (H // 32))
    gl.store(QS + g, rs, g < M * (H // 32))
    gl.store(QS + M * H // 32 + g, ss, g < M * (H // 32))

@gluon.jit
def _router_linear(X, W, L, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, Counts, pid_m, pid_n, WARPS: gl.constexpr=1, INIT_SHARDS: gl.constexpr=8):
    init_shard = pid_m * (256 // BN) + pid_n
    if init_shard < INIT_SHARDS:
        counter = gl.arange(0, 256, gl.BlockedLayout([1], [64], [WARPS], [0]))
        gl.store(Counts + init_shard * 256 + counter, 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1])
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WARPS, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, WARPS], [1, 0]) if M > 128 else gl.BlockedLayout([8, 1], [16, 4], [1, WARPS], [0, 1])
    mi = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    ni = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, bl))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for base in range(H // BK):
        a = gl.load(X + mi[:, None] * SX + (base * BK + ak)[None, :], mi[:, None] < M, 0)
        b = gl.load(W + ni[None, :] * H + (base * BK + bk)[:, None])
        acc = gl.amd.cdna4.mfma(gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8)), gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8), assert_trivial=M > 128), acc)
    mm = pid_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    nn = pid_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(L + mm[:, None] * 256 + nn[None, :], acc, mm[:, None] < M)

@gluon.jit
def _router_and_quantize(X, W, L, Q, QS, Counts, M: gl.constexpr, H: gl.constexpr, SX: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, GROUPS: gl.constexpr, WARPS: gl.constexpr, SHARDS: gl.constexpr):
    ROUTER_ROWS: gl.constexpr = triton.cdiv(M, BM)
    ROUTER_CTAS: gl.constexpr = ROUTER_ROWS * (256 // BN)
    pid = gl.program_id(0)
    if pid < ROUTER_CTAS:
        _router_linear(X, W, L, M, H, SX, BM, BN, BK, Counts, pid % ROUTER_ROWS, pid // ROUTER_ROWS, WARPS, SHARDS)
    else:
        _quantize_input(X, Q, QS, M, H, SX, GROUPS, ROUTER_CTAS, WARPS)

@gluon.jit
def _select_routes(L, Bias, Records, Counts, SHARDS: gl.constexpr, TICKET_STRIDE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    m = gl.program_id(0)
    e = gl.arange(0, 256, layout)
    prob = 1.0 / (1.0 + gl.exp(-gl.load(L + m * 256 + e).to(gl.float32)))
    score = prob + gl.load(Bias + e).to(gl.float32)
    available = gl.full((256,), True, gl.int1, layout)
    selected_prob = gl.full((256,), 0.0, gl.float32, layout)
    selected_id = gl.full((256,), 0, gl.int32, layout)
    total = 0.0
    for j in gl.static_range(8):
        maximum = gl.max(score, 0)
        idx = gl.min(gl.where(available & (score == maximum), e, 256), 0)
        idx = gl.where(idx < 256, idx, gl.min(gl.where(available, e, 256), 0))
        p = gl.sum(gl.gather(prob, gl.full((1,), idx, gl.int32, layout), 0), 0)
        total += p
        selected_prob = gl.where(e == j, p, selected_prob)
        selected_id = gl.where(e == j, idx, selected_id)
        available &= e != idx
        score = gl.where(e == idx, -float('inf'), score)
    ticket = gl.atomic_add(Counts + m // 32 % SHARDS * 256 + selected_id, 1, e < 8, sem='relaxed')
    weight = selected_prob / total * 2.5
    record = (selected_id * TICKET_STRIDE + ticket).to(gl.uint64)
    record |= weight.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
    gl.store(Records + m * 8 + e, record, e < 8)

@gluon.jit
def _pack_job(expert, live, block):
    return expert | live << 9 | block << 17

@gluon.jit
def _down_job_counts(counts, BM: gl.constexpr, WIDE: gl.constexpr):
    remainder = counts % BM
    return counts // BM * 2 + gl.where(remainder > WIDE, 2, (remainder > 0).to(gl.int32))

@gluon.jit
def _write_height_descriptors(UpInfo, e, experts, counts, offset, count, M: gl.constexpr, BM: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
    live = gl.minimum(BM, count - b * BM)
    remainder = counts % BM
    destination = gl.full(b.shape, 0, gl.int32, layout)
    for level in gl.static_range(3 + (BM == 128)):
        height = gl.constexpr(BM >> level)
        lower = gl.constexpr(0 if height == 16 else height // 2)
        same = ((remainder > lower) & (remainder <= height)).to(gl.int32)
        if height == BM:
            same += counts // BM
            before = gl.sum(gl.where(experts < e, same, 0), 0)
            local = b
        else:
            higher = counts // BM + (remainder > height).to(gl.int32)
            shared_higher = gl.constexpr(M // BM + (M % BM > height))
            before = gl.sum(higher, 0) + shared_higher + gl.sum(gl.where(experts < e, same, 0), 0)
            local = gl.full(b.shape, 0, gl.int32, layout)
        destination = gl.where((live > lower) & (live <= height), before + local, destination)
    descriptor = _pack_job(e, live, offset + b)
    gl.store(UpInfo + destination, descriptor, b < gl.cdiv(count, BM))

@gluon.jit
def _sum_shard_counts(Counts, SHARDS: gl.constexpr, before=0, WITH_PREFIX: gl.constexpr=False):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, 1], [1, 0])
    shard = gl.arange(0, SHARDS, gl.SliceLayout(1, layout))
    expert = gl.arange(0, 256, gl.SliceLayout(0, layout))
    partial = gl.load(Counts + shard[:, None] * 256 + expert[None, :])
    native: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    totals = gl.convert_layout(gl.sum(partial, 0), native)
    if WITH_PREFIX:
        prefix = gl.sum(gl.where(shard[:, None] < before, partial, 0), 0)
        return (totals, gl.convert_layout(prefix, native))
    else:
        return totals

@gluon.jit
def _write_down_tiles(Jobs, e, experts, counts, offset, count, arena_start, M: gl.constexpr, BM: gl.constexpr, WIDE: gl.constexpr=32):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    sizes = _down_job_counts(counts, BM, WIDE)
    start = gl.sum(gl.where(experts < e, sizes, 0), 0)
    b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
    live = gl.minimum(BM, count - b * BM)
    descriptor = _pack_job(e, live, offset + b).to(gl.uint64)
    descriptor |= (arena_start + b * BM).to(gl.uint64) << 32
    destination = start + b * 2
    if M == 128:
        tall_sizes = (counts // BM + (counts % BM > WIDE).to(gl.int32)) * 2
        shared_tall: gl.constexpr = (M // BM + (M % BM > WIDE)) * 2
        tall_before = gl.sum(gl.where(experts < e, tall_sizes, 0), 0)
        short_before = gl.sum(gl.where(experts < e, sizes - tall_sizes, 0), 0)
        tall_total = gl.sum(tall_sizes, 0) + shared_tall
        destination = gl.where(live > WIDE, tall_before + b * 2, tall_total + short_before)
    gl.store(Jobs + destination, descriptor, b < gl.cdiv(count, BM))
    gl.store(Jobs + destination + 1, descriptor | 1 << 29, (b < gl.cdiv(count, BM)) & (live > WIDE))

@gluon.jit
def _scatter_routes(Codes, Counts, Sorted, chunk, M: gl.constexpr, BM: gl.constexpr, SHARDS: gl.constexpr, TICKET_STRIDE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    lane = gl.arange(0, 256, layout)
    route = chunk * 256 + lane
    code = gl.load(Codes + route, route < M * 8, 0).to(gl.int32)
    expert, ticket = (code // TICKET_STRIDE, code % TICKET_STRIDE)
    counts, prefix_counts = _sum_shard_counts(Counts, SHARDS, chunk % SHARDS, True)
    tiles = gl.cdiv(counts, BM)
    offsets = (gl.associative_scan(tiles, 0, _add) - tiles) * BM
    offset = gl.gather(offsets, expert, 0)
    prefix = gl.gather(prefix_counts, expert, 0)
    gl.store(Sorted + offset + prefix + ticket, route, route < M * 8)

@gluon.jit
def _prepare_tickets(Codes, Counts, Sorted, UpInfo, Jobs, M: gl.constexpr, CHUNKS: gl.constexpr, BM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, SCHEDULED: gl.constexpr, DOWN_SCHEDULED: gl.constexpr, WIDE: gl.constexpr=32, SHARDS: gl.constexpr=8, TICKET_STRIDE: gl.constexpr=1024):
    pid = gl.program_id(0)
    if pid < CHUNKS:
        _scatter_routes(Codes, Counts, Sorted, pid, M, BM, SHARDS, TICKET_STRIDE)
    else:
        e = pid - CHUNKS
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
        experts = gl.arange(0, 256, layout)
        counts = _sum_shard_counts(Counts, SHARDS)
        tiles = gl.cdiv(counts, BM)
        if e == 257:
            active = gl.sum(tiles, 0)
            hole = gl.arange(0, _artifact_next_power_of_2(SCHEDULED), layout)
            gl.store(UpInfo + hole, 0, (hole >= active + triton.cdiv(M, BM)) & (hole < SCHEDULED))
            sizes = _down_job_counts(counts, BM, WIDE)
            shared_tiles: gl.constexpr = M // BM * 2 + (2 if M % BM > WIDE else 1 if M % BM > 0 else 0)
            active_tiles = gl.sum(sizes, 0) + shared_tiles
            holes = gl.arange(0, _artifact_next_power_of_2(DOWN_SCHEDULED), layout)
            gl.store(Jobs + holes, 0, (holes >= active_tiles) & (holes < DOWN_SCHEDULED))
        else:
            if e < 256:
                offset = gl.sum(gl.where(experts < e, tiles, 0), 0)
                count = gl.sum(gl.where(experts == e, counts, 0), 0)
            else:
                offset = ROUTED_BLOCKS
                count = M
            _write_height_descriptors(UpInfo, e, experts, counts, offset, count, M, BM)
            remainder = counts % BM
            tail = gl.where(remainder > 64, 128, gl.where(remainder > 32, 64, gl.where(remainder > 16, 32, gl.where(remainder > 0, 16, 0))))
            arena_sizes = counts // BM * BM + tail
            arena_start = gl.sum(gl.where(experts < e, arena_sizes, 0), 0)
            _write_down_tiles(Jobs, e, experts, counts, offset, count, arena_start, M, BM, WIDE)
            b = gl.arange(0, _artifact_next_power_of_2(triton.cdiv(M, BM)), layout)
            gl.store(Sorted + ROUTED_BLOCKS * BM + offset + b, arena_start + b * BM, b < gl.cdiv(count, BM))

@gluon.jit
def _unpack_u32(words):
    b0 = words.to(gl.uint8)
    b1 = (words >> 8).to(gl.uint8)
    b2 = (words >> 16).to(gl.uint8)
    b3 = (words >> 24).to(gl.uint8)
    return gl.join(gl.join(b0, b2), gl.join(b1, b3))

@gluon.jit
def _word_bytes(words):
    return _unpack_u32(words).reshape((words.shape[0], words.shape[1] * 4))

@gluon.jit
def _weight_offset(n, k, K: gl.constexpr):
    byte = k // 2
    return (((n // 16 * (K // 64) + byte // 32) * 2 + byte // 16 % 2) * 16 + n % 16) * 16 + byte % 16

@gluon.jit
def _load_packed_weight(W, S, expert, column, base, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, CACHE: gl.constexpr='', NATIVE: gl.constexpr=False, REBASE: gl.constexpr=False):
    W = W + expert * (N * K // 2)
    S = S + expert * (N * (triton.cdiv(K // 32, 8) * 8))
    if REBASE:
        W += column * (BN // 2) * (K // 2)
        S += column * (BN // 2) * (triton.cdiv(K // 32, 8) * 8)
        column = 0
    packed: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [0, 1]) if NATIVE else gl.BlockedLayout([1, 4], [32, 2], [2, 2] if BK == 512 else [4, 1], [1, 0])
    n = gl.arange(0, BN, gl.SliceLayout(1, packed))
    if UP:
        n = column * (BN // 2) + n % (BN // 2) + n // (BN // 2) * (N // 2)
    else:
        n = column * BN + n
    k = base * BK + 8 * gl.arange(0, BK // 8, gl.SliceLayout(0, packed))
    offset = _weight_offset(n[:, None], k[None, :], K) // 4
    words = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), offset, cache=CACHE)
    sl: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    idx = gl.arange(0, BN * BK // 128, sl)
    nb = idx // (BK // 4)
    if UP:
        nb = column * (BN // 64) + nb % (BN // 64) + nb // (BN // 64) * (N // 64)
    else:
        nb = column * (BN // 32) + nb
    kg = idx // 64 % (BK // 256)
    inner = idx % 64
    sw = gl.amd.cdna4.buffer_load(S.to(gl.pointer_type(gl.uint32)), nb * (K // 4) + base * (BK // 4) + kg * 64 + inner)
    raw = sw.reshape((BN // 32, BK // 256, 4, 16))
    packed_scales = _unpack_u32(raw)
    scale_byte = gl.permute(packed_scales, (0, 5, 3, 1, 4, 2)).reshape((BN, BK // 32))
    return (words, scale_byte)

@gluon.jit
def _native_down_gemm(X, XS, W, WS, row, expert, column, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr):
    BK: gl.constexpr = 256
    mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4])
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [16, BK // 32])
    bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
    native: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [0, 1])
    ar = gl.convert_layout(row, gl.SliceLayout(1, native))
    ak = gl.arange(0, BK // 8, gl.SliceLayout(0, native))
    bn = column * BN + gl.arange(0, BN, gl.SliceLayout(1, native))
    asr = gl.convert_layout(row, gl.SliceLayout(1, asl))
    ask = gl.arange(0, BK // 32, gl.SliceLayout(0, asl))
    bsn = column * BN + gl.arange(0, BN, gl.SliceLayout(1, bsl))
    bsk = gl.arange(0, BK // 32, gl.SliceLayout(0, bsl))
    W += expert * (N * K // 2)
    WS += expert * (N * K // 32)
    acc = gl.zeros((16, BN), gl.float32, mma)
    for base in gl.static_range(K // BK):
        aw = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), ar[:, None] * (K // 8) + base * (BK // 8) + ak[None, :])
        bw = gl.amd.cdna4.buffer_load(W.to(gl.pointer_type(gl.uint32)), _weight_offset(bn[:, None], base * BK + 8 * ak[None, :], K) // 4)
        a = gl.convert_layout(_word_bytes(aw), ad, assert_trivial=True)
        b = gl.convert_layout(_word_bytes(bw).T, bd, assert_trivial=True)
        sa_offset = asr[:, None] * (K // 32) + base * (BK // 32) + ask[None, :]
        sa_word = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), sa_offset // 4)
        sa = (sa_word >> sa_offset % 4 * 8).to(gl.uint8)
        kg = base * (BK // 32) + bsk
        scale_offset = bsn[:, None] // 32 * K + kg[None, :] // 8 * 256 + kg[None, :] % 4 * 64 + bsn[:, None] % 16 * 4 + kg[None, :] // 4 % 2 * 2 + bsn[:, None] // 16 % 2
        sb_word = gl.amd.cdna4.buffer_load(WS.to(gl.pointer_type(gl.uint32)), scale_offset // 4)
        sb = (sb_word >> scale_offset % 4 * 8).to(gl.uint8)
        acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
    return acc

@gluon.jit
def _mxfp4_gemm(X, XS, W, WS, row, expert, column, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, UP: gl.constexpr, TM: gl.constexpr):
    if not UP and TM == 16:
        return _native_down_gemm(X, XS, W, WS, row, expert, column, N, K, BN)
    else:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4] if TM <= 32 else [2, 2])
        ad: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
        bd: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
        asl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(ad, [TM, BK // 32])
        bsl: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(bd, [BN, BK // 32])
        al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [2, 2] if BK == 512 else [4, 1], [1, 0])
        ki = gl.arange(0, BK // 8, gl.SliceLayout(0, al))
        a_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(ad, [TM, BK // 2], gl.uint8) if M == 128 and (TM >= 32 or BK >= 512) or (M > 128 and (not UP) and (TM >= 32)) else gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
        b_shared_layout: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(bd, [BK // 2, BN], gl.uint8) if M == 128 or not UP else gl.SwizzledSharedLayout(16, 1, 8, [0, 1])
        a_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 2], a_shared_layout)
        NATIVE_B: gl.constexpr = TM <= 32 and (M == 128 or UP)
        if not NATIVE_B:
            b_shared = gl.allocate_shared_memory(gl.uint8, [BK // 2, BN], b_shared_layout)
        bs_shared_layout: gl.constexpr = gl.SharedLinearLayout([[0, 4], [16, 0], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [32, 0], [64, 0]] + ([[128, 0]] if BN == 256 else []) + ([[0, 8]] if BK >= 512 else []) + ([[0, 16]] if BK >= 1024 else []))
        bs_shared = gl.allocate_shared_memory(gl.uint8, [BN, BK // 32], bs_shared_layout)
        as_shared_layout: gl.constexpr = gl.SharedLinearLayout([[0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2], [0, 16]]) if M == 128 and UP and (BK == 1024) else gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        as_shared = gl.allocate_shared_memory(gl.uint8, [TM, BK // 32], as_shared_layout)
        as_load_layout: gl.constexpr = gl.BlockedLayout([1, 2], [32, 2], [4, 1], [1, 0])
        scale_row = gl.convert_layout(row, gl.SliceLayout(1, as_load_layout))
        scale_word = gl.arange(0, BK // 128, gl.SliceLayout(0, as_load_layout))
        acc = gl.zeros((TM, BN), gl.float32, mma)
        WEIGHT_CACHE: gl.constexpr = '' if M > 128 and UP and (TM == 128) else '.cg'
        UNROLL: gl.constexpr = 2 if not UP and (M > 128 or TM <= 32) and (K % (BK * 2) == 0) else 1
        for group in range(K // (BK * UNROLL)):
            for offset in gl.static_range(UNROLL):
                base = group * UNROLL + offset
                a_words = gl.amd.cdna4.buffer_load(X.to(gl.pointer_type(gl.uint32)), row[:, None] * (K // 8) + base * (BK // 8) + ki[None, :])
                a = _word_bytes(a_words)
                sa_words = gl.amd.cdna4.buffer_load(XS.to(gl.pointer_type(gl.uint32)), scale_row[:, None] * (K // 128) + base * (BK // 128) + scale_word[None, :])
                sa_bytes = _word_bytes(sa_words)
                words, scales = _load_packed_weight(W, WS, expert, column, base, N, K, BN, BK, UP, WEIGHT_CACHE, NATIVE_B, M > 128 and UP)
                b = _word_bytes(words).T
                a_shared.store(a)
                if not NATIVE_B:
                    b_shared.store(b)
                bs_shared.store(scales)
                as_shared.store(sa_bytes)
                sa = as_shared.load(asl)
                a = a_shared.load(ad)
                if NATIVE_B:
                    b = gl.convert_layout(b, bd, assert_trivial=True)
                else:
                    b = b_shared.load(bd)
                sb = bs_shared.load(bsl)
                acc = gl.amd.cdna4.mfma_scaled(a, sa, 'e2m1', b, sb, 'e2m1', acc)
        return acc

@gluon.jit
def _activate_and_store(acc, Q, QS, arena_row, column, shared, N: gl.constexpr, BN: gl.constexpr, TM: gl.constexpr, M: gl.constexpr):
    gate, up = gl.split(gl.permute(gl.reshape(acc, (TM, 2, BN // 2)), (0, 2, 1)))
    if shared:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
    activated = (gate * (1.0 / (1.0 + gl.exp(-gate))) * up).to(gl.bfloat16)
    ep: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    if M > 128:
        activated = gl.convert_layout(activated, ep).to(gl.float32)
    else:
        activated = gl.convert_layout(activated.to(gl.float32), ep)
    codes, scales = _encode_groups(gl.reshape(activated, (TM * (BN // 64), 32)), shared)
    codes = codes.reshape((TM, BN // 4))
    scales = scales.reshape((TM, BN // 64))
    rr = gl.arange(0, TM, gl.SliceLayout(1, codes.type.layout))
    nn = column * (BN // 4) + gl.arange(0, BN // 4, gl.SliceLayout(0, codes.type.layout))
    gl.store(Q + (arena_row + rr[:, None]) * (N // 4) + nn[None, :], codes)
    rr_s = gl.arange(0, TM, gl.SliceLayout(1, scales.type.layout))
    nn_s = column * (BN // 64) + gl.arange(0, BN // 64, gl.SliceLayout(0, scales.type.layout))
    gl.store(QS + (arena_row + rr_s[:, None]) * (N // 64) + nn_s[None, :], scales)

@gluon.jit
def _store_down(acc, Parts, Y, Sorted, block, column, shared, live, M: gl.constexpr, N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, TM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr):
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
    rr = gl.arange(0, TM, gl.SliceLayout(1, store_layout))
    nn = column * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    if shared:
        token = (block - ROUTED_BLOCKS) * BM + rr
        gl.store(Y + token[:, None] * N + nn[None, :], result, rr[:, None] < live)
    else:
        route = gl.load(Sorted + block * BM + rr, rr < live, 0)
        PANEL: gl.constexpr = 256 if M > 128 else 128
        part_base = Parts + column * BN // PANEL * M * 8 * PANEL
        address = route[:, None] * PANEL + nn[None, :] % PANEL
        if BN > PANEL:
            address += nn[None, :] % BN // PANEL * M * 8 * PANEL
        gl.amd.cdna4.buffer_store(result.to(Parts.dtype.element_ty), part_base, address, rr[:, None] < live)

@gluon.jit
def _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, UP: gl.constexpr, TM: gl.constexpr):
    al: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [2, 2] if BK == 512 else [4, 1], [1, 0])
    mi = gl.arange(0, TM, gl.SliceLayout(1, al))
    shared = expert == 256
    if UP:
        arena_row = gl.load(Sorted + ROUTED_BLOCKS * BM + block)
        if shared:
            route = ((block - ROUTED_BLOCKS) * BM + mi) * 8
        else:
            route = gl.load(Sorted + block * BM + mi, mi < live, 0)
        row = gl.where(mi < live, route // 8 + gl.where(shared, M, 0), 0)
    else:
        arena_row = packed_rows.to(gl.int32)
        row = arena_row + mi
    acc = _mxfp4_gemm(X, XS, W, WS, row, expert, column, M, N, K, BN, BK, UP, TM)
    if UP:
        _activate_and_store(acc, Q, QS, arena_row, column, shared, N, BN, TM, M)
    else:
        _store_down(acc, Parts, Y, Sorted, block, column, shared, live, M, N, BM, BN, TM, ROUTED_BLOCKS)

@gluon.jit
def _scaled_experts(X, XS, W, WS, Sorted, Info, Q, QS, Parts, Y, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, BM: gl.constexpr, ROUTED_BLOCKS: gl.constexpr, GROUP: gl.constexpr, UP: gl.constexpr, UP_N: gl.constexpr=128):
    pid = gl.program_id(0)
    COLS: gl.constexpr = N // (UP_N if UP else 256)
    tile = pid // (GROUP * COLS) * GROUP + pid % GROUP
    panel = pid // GROUP % COLS
    descriptor = gl.load(Info + tile)
    info = descriptor.to(gl.int32)
    packed_rows = (descriptor.to(gl.uint64) >> 32).to(gl.uint32)
    if info != 0:
        expert = info & 511
        live = info >> 9 & 255
        block = info >> 17 & 4095
        column = panel if UP else panel * 2 + (info >> 29)
        SHORT_N: gl.constexpr = UP_N if UP else 256
        SHORT_K: gl.constexpr = min(1024, K & -K) if UP else 256
        MEDIUM_K: gl.constexpr = 512 if UP and M > 128 else 256
        TALL_N: gl.constexpr = UP_N if UP else 128
        if live <= 16:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, SHORT_N, SHORT_K, ROUTED_BLOCKS, UP, 16)
        elif live <= 32:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, SHORT_N, MEDIUM_K, ROUTED_BLOCKS, UP, 32)
        elif live <= 64:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, panel, expert, live, packed_rows, M, N, K, BM, SHORT_N, 256, ROUTED_BLOCKS, UP, 64)
        else:
            _scaled_tile(X, XS, W, WS, Sorted, Q, QS, Parts, Y, block, column, expert, live, packed_rows, M, N, K, BM, TALL_N, 256, ROUTED_BLOCKS, UP, BM)

@gluon.jit
def _reduce_parts(P, Y, Records, M: gl.constexpr, H: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr, CACHE: gl.constexpr='', VECTOR: gl.constexpr=1):
    gl.static_assert(BLOCK % 128 == 0)
    m = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([VECTOR], [64], [WARPS], [0])
    inner = gl.arange(0, BLOCK, layout)
    h = gl.program_id(1) * BLOCK + inner
    PANEL: gl.constexpr = 256 if M > 128 else 128
    part_base = P + gl.program_id(1) * BLOCK // PANEL * M * 8 * PANEL
    value = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for rank in gl.static_range(8):
        record = gl.load(Records + m * 8 + rank).to(gl.uint64)
        route = m * 8 + rank
        weight = (record >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
        address = inner // PANEL * M * 8 * PANEL + route * PANEL + h % PANEL
        contribution = gl.amd.cdna4.buffer_load(part_base, address, cache=CACHE).to(gl.float32)
        value += contribution * weight
    value += gl.load(Y + m * H + h).to(gl.float32)
    gl.store(Y + m * H + h, value)

def fused_moe(x, router, correction_bias, w13, w13_scale, w2, w2_scale):
    m, h = x.shape
    intermediate = w13.shape[1] // 2
    block_m = 128
    routed_blocks = triton.cdiv(m * 8, block_m) + 256
    blocks = routed_blocks + triton.cdiv(m, block_m)
    chunks = triton.cdiv(m * 8, 256)
    shards = 8
    ticket_stride = _artifact_next_power_of_2(triton.cdiv(m, shards * 32) * 32)
    scheduled_blocks = triton.cdiv(blocks, 8) * 8
    router_n = 16
    router_m = 8 if m == 128 else 16
    router_k = min(h & -h, 1024)
    router_warps = 1
    group_up = 4 if m == 128 else 8
    up_n = 128
    group_down = 2
    wide_down = 64
    down_blocks = triton.cdiv(triton.cdiv(m * 8, 64) + 256 + triton.cdiv(m, 64), 8) * 8

    def empty(shape, dtype=torch.bfloat16):
        return torch.empty(shape, device=x.device, dtype=dtype)
    part_width = 128 if m == 128 else 256
    parts = empty((h // part_width, m * 8, part_width), torch.bfloat16)
    xq = empty((2 * m, h // 2), torch.uint8)
    xs = empty((2 * m, h // 32), torch.uint8)
    logits = empty((m, 256))
    records = empty((m, 8), torch.int64)
    partial_counts = empty((shards, 256), torch.int32)
    up_info = empty((scheduled_blocks,), torch.int32)
    jobs = empty((down_blocks,), torch.int64)
    sorted_routes = empty((routed_blocks * block_m + blocks,), torch.int32)
    arena_rows = m * 8 + 256 * 64 + triton.cdiv(m, block_m) * block_m
    aq = empty((arena_rows, intermediate // 2), torch.uint8)
    aqs = empty((arena_rows, intermediate // 32), torch.uint8)
    out = empty((m, h))
    quant_groups = 128
    quant_ctas = triton.cdiv(m * (h // 32), quant_groups)
    router_ctas = triton.cdiv(m, router_m) * (256 // router_n)
    _router_and_quantize[router_ctas + quant_ctas,](x, router, logits, xq, xs, partial_counts, m, h, x.stride(0), router_m, router_n, router_k, quant_groups, router_warps, shards, num_warps=router_warps, enable_fp_fusion=False)
    _select_routes[m,](logits, correction_bias, records, partial_counts, shards, ticket_stride, num_warps=1, enable_fp_fusion=False)
    _prepare_tickets[chunks + 258,](records, partial_counts, sorted_routes, up_info, jobs, m, chunks, block_m, routed_blocks, scheduled_blocks, down_blocks, wide_down, shards, ticket_stride, num_warps=1, enable_fp_fusion=False)
    up_columns = 2 * intermediate // up_n
    _scaled_experts[scheduled_blocks * up_columns,](xq, xs, w13, w13_scale, sorted_routes, up_info, aq, aqs, parts, out, m, 2 * intermediate, h, block_m, routed_blocks, group_up, True, up_n, enable_fp_fusion=False)
    _scaled_experts[down_blocks * (h // 256),](aq, aqs, w2, w2_scale, sorted_routes, jobs, aq, aqs, parts, out, m, h, intermediate, block_m, routed_blocks, group_down, False)
    reduce_block = 512 if m == 128 else 256
    reduce_warps = 4 if m == 128 else 1
    _reduce_parts[m, h // reduce_block](parts, out, records, m, h, reduce_block, reduce_warps, '.cg', 2, num_warps=reduce_warps, enable_fp_fusion=False)
    return out
