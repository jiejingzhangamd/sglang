## Tuning Triton MoE Kernels

This directory contains benchmarking tools for MoE (Mixture of Experts) kernels.

### Overview

The tuning tools support both **Tensor Parallelism (TP)** and **Expert Parallelism (EP)** modes:

- **TP Mode**: Traditional tensor parallelism where intermediate layers are sharded across GPUs
- **EP Mode**: Expert parallelism where experts are distributed across GPUs. Can be combined with TP mode (e.g., `--tp-size 8 --ep-size 2`)
- **MLLM Support**: Multi-modal Large Language Models with text encoders (e.g., Llama4, Qwen3VL)

### Tuning Tools

#### 1. `tuning_fused_moe_triton.py`
A unified tool for tuning the `fused_moe_triton` kernel. Adapted from [vllm's benchmark_moe.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py), with support for EP mode and various model architectures.

#### 2. `tuning_fused_moe_triton_sep.py`
A specialized tool for separate kernel tuning, optimizing the first and second MoE kernels independently with TMA (Tensor Memory Accelerator) support.

### Usage Examples

#### Basic TP Mode Tuning
```bash
# Tune Mixtral-8x7B with default TP settings
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tune

# Tune Qwen2-57B with FP8 and TP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 4 \
    --dtype fp8_w8a8 \
    --tune

# Tune DeepSeek-V3 with FP8 and TP=8
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8 \
    --dtype fp8_w8a8 \
    --tune
```

#### EP Mode Tuning (Expert Parallelism)
**Note**: EP mode can be used alone or combined with TP mode. When using both, ensure `tp_size` is divisible by `ep_size`.

```bash
# Tune Mixtral-8x7B with EP=2 only
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --ep-size 2 \
    --tune

# Tune Qwen2-57B with TP=8 and EP=4 (combined mode)
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 8 \
    --ep-size 4 \
    --dtype fp8_w8a8 \
    --tune
```

#### MLLM Model Tuning (Multi-modal)
```bash
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tp-size 2 \
    --tune
```

#### Separate Kernel Tuning with `tuning_fused_moe_triton_sep.py`

This tool requires pre-generated topk_ids files and supports both TP and EP modes:

Edit the code file (such as srt/models/deepseek_v2.py) in the Python site package and add the logic for saving topk_ids:

```python
# import get_tensor_model_parallel_rank
# DeepseekV2MoE::forward_normal
if hidden_states.shape[0] >= 4096 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = xxxx
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(topk_output.topk_ids, f"{topk_ids_dir}/topk_ids_layer{self.layer_id}_idx{self.save_idx}.pt")
    self.save_idx += 1
```

Launch sglang server and send request using `benchmark/kernels/fused_moe_triton/tuning_client.py`
```bash
python benchmark/kernels/fused_moe_triton/tuning_client.py --port 8000
```

```bash
# TP Mode: Tune separate kernels with TP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --tp-size 4 \
    --topk-ids-dir /path/to/topk_ids \
    --tune

# EP Mode: Tune separate kernels with TP=4 and EP=2
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 4 \
    --ep-size 2 \
    --topk-ids-dir /path/to/topk_ids \
    --tune

# MLLM: Tune DeepSeek-V3 with separate kernels, TP=8 and EP=4
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8 \
    --ep-size 4 \
    --dtype fp8_w8a8 \
    --topk-ids-dir /path/to/topk_ids \
    --tune

# Benchmark specific config without tuning
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 4 \
    --batch-size 1024 \
    --dtype fp8_w8a8 \
    --configs 128 256 128 16 8 4 \
    --topk-ids-dir /path/to/topk_ids
```

#### Advanced Options
```bash
# Channel-wise quantization
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model meituan/DeepSeek-R1-Channel-INT8 \
    --tp-size 16 \
    --dtype int8_w8a8 \
    --per-channel-quant \
    --tune

# Specific batch size tuning
python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --batch-size 2048 \
    --tune
```

### Configuration Files

After tuning, configuration files will be generated:
- **Standard tuning**: `E=64,N=640,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json`
- **Separate kernel tuning**: Two files for up/down kernels with TMA optimization flags

Move these files to `sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_version/` directory to use them in SGLang.

### Supported Models

- **Mixtral**: mistralai/Mixtral-8x7B-Instruct-v0.1, mixtral-8x22b
- **Qwen**: Qwen2-57B, Qwen3-235B, Qwen3VL (MLLM)
- **DeepSeek**: DeepSeek-V2, DeepSeek-V3, DeepSeek-R1
- **Llama**: Llama4-Vision (MLLM)
- **DBRX**: databricks/dbrx-instruct
- **Jamba**: ai21labs/AI21-Jamba
- **Grok**: xai-org/grok-1
- **GLM**: THUDM/glm-4-9b-chat
- **Bailing**: Custom MoE models

### Parameters Reference

- `--model`: HuggingFace model name or local path
- `--tp-size`: Tensor parallelism size (default: 2)
- `--ep-size`: Expert parallelism size (default: 1, can be combined with TP mode, ensure tp_size is divisible by ep_size)
- `--dtype`: Data type (`auto`, `fp8_w8a8`, `int8_w8a16`, `int8_w8a8`)
- `--batch-size`: Specific batch size for tuning (optional)
- `--tune`: Enable tuning mode
- `--per-channel-quant`: Enable per-channel quantization
- `--disable-shared-experts-fusion`: Disable shared expert fusion for some models
- `--topk-ids-dir`: Directory containing pre-generated topk_ids (for sep tool only)
- `--configs`: Manual config specification [BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages]

### Performance Comparison Tool

- `benchmark_vllm_vs_sglang_fused_moe_triton.py`: A tool for comparing the performance of fused MoE kernels between vllm and sglang implementations. Supports various model architectures and data types.

Example usage:
```bash
# Compare with default settings (Mixtral model)
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py

# Compare with FP8 mode for Qwen2-57B
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model Qwen/Qwen2-57B-A14B-Instruct \
    --use-fp8-w8a8

# Compare with custom TP size
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8

# Compare with custom TP size
python benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --tp-size 8
```

The benchmark results will be saved as plots and data files in the specified output directory (default: `./configs/benchmark_ops/vllm_sglang_fused_moe/`).

- `benchmark_torch_compile_fused_moe.py`: A tool for benchmarking the performance of the fused MoE kernel with `torch.compile` and original fused MoE kernel.

Usage is similar to `benchmark_vllm_vs_sglang_fused_moe_triton.py`, note that `torch.compile` does not support `fp8_w8a8` and `int8_w8a8` fused_moe_kernel. Both tools now support EP mode with `--ep-size` parameter.

### Server Trace A/B Tool

`profile_moe_server_ab.py` measures the aggregate GPU-time change of a MoE-only
server modification. It captures the same deterministic decode window from a
baseline and candidate server, then compares the rank-local GPU activity. Keep
all server arguments identical except for the MoE implementation under test.

Capture the baseline while its server is running:

```bash
python benchmark/kernels/fused_moe_triton/profile_moe_server_ab.py capture \
  --url http://127.0.0.1:30000 \
  --output-dir /shared/profiles/native \
  --profile-id native \
  --steps 50
```

Restart the server with the candidate MoE implementation and capture it:

```bash
python benchmark/kernels/fused_moe_triton/profile_moe_server_ab.py capture \
  --url http://127.0.0.1:30000 \
  --output-dir /shared/profiles/candidate \
  --profile-id candidate \
  --steps 50
```

If the client and server see different filesystem paths, pass the path visible
to the server with `--server-output-dir`. The measurement request must be long
enough to remain active for all requested profiling steps.

Compare the rank-0 traces. For speculative decoding, pass the measured average
accepted tokens per server step to convert the result to milliseconds per
output token:

```bash
python benchmark/kernels/fused_moe_triton/profile_moe_server_ab.py compare \
  --baseline-dir /shared/profiles/native \
  --candidate-dir /shared/profiles/candidate \
  --steps 50 \
  --accepted-tokens-per-step 3.61 \
  --target-ms-per-output-token 1.12 \
  --output /shared/profiles/report.json
```

The report includes summed GPU activity, the union of overlapping GPU activity,
the per-step and per-output-token deltas, the largest GPU activities, and a
deterministic output-text hash check. The delta is attributable to MoE only when
MoE is the sole server difference; the tool does not infer operator ownership
from kernel names.

### GLM-5.2 Triton Gluon TP4 and TP8 kernel snapshots

`glm52_triton_gluon_tp4/` and `glm52_triton_gluon_tp8/` contain the consolidated
gfx950 MXFP4 fused-MoE implementations. The TP4 profile has 48 target and
MTP/draft dispatch entries backed by four source files. The TP8 profile has 64
entries backed by eight source files. Relative to the per-shape snapshot, TP4
uses 4 instead of 20 source files and 2,086 instead of 10,576 kernel lines
(-80.3%). TP8 uses 8 instead of 18 source files and 3,772 instead of 7,966
kernel lines (-52.6%).
Related active-batch shapes share a kernel implementation, while tile sizes,
warp counts, splits, grouping, and other shape-specific choices remain static
or `gl.constexpr` values so Triton specializes them at compile time. The compact
schema-v3 profiles record common source, SHA-256, and semantics once per shape
family. Use `profile_schema.load_profile()` (or run `profile_schema.py` as a
CLI) to expand them into the flat `specializations` rows accepted by existing
schema-v2 consumers. TP8 continuously covers every active batch from 1 through
32768 in both target and MTP/draft modes. Exact tuned shapes take precedence;
the remaining shapes dispatch to the corresponding compile-time-specialized
shape family instead of falling back to the native MoE backend. The CPU test
expands both profiles and verifies the recorded source digests and continuous
coverage without importing GPU dependencies.

The same four TP4-family sources also cover total TP8 with EP2, EP4, and EP8.
The runtime passes the rank's global `expert_start`; each kernel derives the
local expert count from its packed weights, maps non-local routes to one zero
sentinel, and emits rank-local routed output. SGLang's existing TP8 post-expert
all-reduce reconstructs the global routed result, while the shared dense expert
stays on its native path. This adds no EP-specific kernel copies. The audited
local layouts are respectively 128 experts × 512 intermediate, 64 × 1024, and
32 × 2048. Strict dispatch profiles continuously cover every M from 1 through
16768 in both target and MTP/draft modes; an uncovered shape is an error, not a
native-MoE fallback.

The compact `m128_4192` source was checked directly against rank-masked AITER
output at the previously uncovered M=65, 96, 192, and 255 shapes. All outputs
were finite. Latency is the slowest rank's median Gluon kernel time; AITER was
used as the numerical oracle, not as a timed baseline in this check.

| Layout | Relative L2 range | Maximum absolute error | Gluon latency range (us) |
|:--|--:|--:|--:|
| TP8/EP2 | 0.004556–0.004568 | 0.000488 | 233.9–271.2 |
| TP8/EP4 | 0.004146–0.004172 | 0.000488 | 313.6–384.0 |
| TP8/EP8 | 0.003849–0.003886 | 0.000977 | 547.2–677.2 |

#### Strict AgentX TP8 EP coverage canary

Each EP layout was also run with the InferenceX PR 3329 AgentX configuration,
concurrency 1, MTP `5/6/1`, a 16384-token prefill chunk, and a 120-second
profiling window. Both target and draft used Gluon under strict dispatch; A2A
and native-MoE fallback were disabled. These are independent coverage canaries,
not a controlled performance comparison between EP layouts.

| Layout | Observed gap-family M | P90 ITV (ms) | Output tok/s | Requests | Reports | Skipped/errors |
|:--|:--|--:|--:|--:|--:|--:|
| TP8/EP2 | 203, 212 | 5.206 | 38.023 | 10 | 16 | 0 / 0 |
| TP8/EP4 | 203 | 5.387 | 34.716 | 9 | 16 | 0 / 0 |
| TP8/EP8 | 204 | 6.579 | 46.191 | 9 | 16 | 0 / 0 |

All three server containers exited successfully. The specialization reports
showed eight target and eight draft ranks, and every observed shape selected a
Gluon family; no strict rejection, native fallback, or GPU fault was present.

#### Strict AgentX TP8/EP1 C=10 A/B

The large-M fix was validated end to end with the GLM-5.2 FP4 AgentX workload
on two MI355X nodes. Every arm used the same SGLang runtime commit, strict
target and draft Gluon binding, MTP `3/4/1`, concurrency 10, a 32768-token
prefill chunk, and a 900-second profiling window. The control used the prior
kernel/profile pack; the candidate used the PR-head pack. The nodes ran
opposite arm orders to control both host and second-arm effects.

| Order | Control P90 ITV | PR-head P90 ITV | P90 ITV change | Control output tok/s | PR-head output tok/s | Output change |
|:--|--:|--:|--:|--:|--:|--:|
| Forward | 56.588 | 81.772 | +44.51% | 365.255 | 495.931 | +35.78% |
| Reverse | 52.439 | 85.361 | +62.78% | 355.119 | 458.535 | +29.12% |
| Node/order-balanced geometric mean | 54.474 | 83.547 | **+53.37%** | 360.151 | 476.867 | **+32.41%** |

The paired P90 ITV improvement across the two orders was **+51.02%**, with a
request-bootstrap 95% confidence interval of **+36.44% to +69.21%** over 1,156
matched requests. Total throughput improved by 6.25%. All four profiling arms
completed with zero request errors, zero native MoE fallbacks, zero skipped
Gluon dispatches, and eight target plus eight draft specialization reports.
MTP acceptance rate (0.865-0.867) and GPU cache hit rate (about 96.3%) were
effectively unchanged.

#### Strict AgentX TP8/EP1 C=1 A/B

The same two-node, opposite-order procedure was repeated at concurrency 1. At
this low concurrency the large-prefill interference fixed by the PR is rare, so
P90 ITV was statistically neutral while output throughput improved.

| Order | Control P90 ITV | PR-head P90 ITV | P90 ITV change | Control output tok/s | PR-head output tok/s | Output change |
|:--|--:|--:|--:|--:|--:|--:|
| Forward | 322.768 | 326.454 | +1.14% | 69.774 | 84.165 | +20.63% |
| Reverse | 325.197 | 326.602 | +0.43% | 72.862 | 78.075 | +7.15% |
| Node/order-balanced geometric mean | 323.980 | 326.528 | **+0.79%** | 71.301 | 81.063 | **+13.69%** |

The paired P90 ITV change was **+0.80%**, with a request-bootstrap 95%
confidence interval of **-1.64% to +3.84%** over 209 matched requests. Total
throughput changed by -1.35%. Every profiling arm had zero request errors; one
control warmup request failed before measurement. The PR-head arms had zero
native MoE fallbacks, zero skipped Gluon dispatches, zero tracebacks, and eight
target plus eight draft specialization reports.

#### Prefill chunk-size sweep

With the fixed PR-head kernels held constant, a second two-node,
opposite-order C=10 A/B compared the 32768-token chunk against 8192. Reducing
the chunk did not improve P90 ITV, so the validated configuration retains
32768.

| Order | 32768 P90 ITV | 8192 P90 ITV | P90 ITV change | 32768 output tok/s | 8192 output tok/s | Output change |
|:--|--:|--:|--:|--:|--:|--:|
| Forward | 85.855 | 80.403 | -6.35% | 488.602 | 511.220 | +4.63% |
| Reverse | 91.456 | 85.162 | -6.88% | 439.407 | 453.591 | +3.23% |
| Node/order-balanced geometric mean | 88.611 | 82.748 | **-6.62%** | 463.352 | 481.544 | **+3.93%** |

The paired P90 ITV change was **-6.53%**, with a request-bootstrap 95%
confidence interval of **-15.57% to +2.58%** over 1,184 matched requests;
total throughput changed by -0.31%. All four profiling arms completed with
zero request errors, zero native MoE fallbacks, zero skipped Gluon dispatches,
and eight target plus eight draft reports. One 8192-token arm logged a timed-out
health probe during the one-time AITER attention build before warmup; it was
not a benchmark request or a profiling failure.

The kernels require an AMD gfx950 GPU and Triton 3.8 Gluon. Each source exports
the `fused_moe` entry point expected by the GLM-5.2 integration; server-side
weight packing remains outside this benchmark bundle. The target path accepts
serialized Quark W4A4 MXFP4 MoE weights. The audited GLM NextN draft path also
accepts its exact BF16 expert ABI and lets the bound GLM backend convert those
weights to the same packed MXFP4 layout. FP8 experts, other BF16 topologies,
and arbitrary online MXFP4 conversion are rejected explicitly. Select the
strict backend with
`--moe-runner-backend gluon`; an out-of-tree implementation subclasses
`GluonMoeBackend` and binds through `DeepseekV2MoE.bind_gluon_moe_backend()`.
The backend returns the rank-local routed-plus-shared output, while SGLang
retains the native post-expert collective. A missing implementation, unsupported
layer/call shape, or invalid output is an error; this backend never silently
falls back to native MoE. This directory is a reproducible
kernel snapshot for the A/B benchmark, not a replacement for SGLang's native
MoE dispatch.

The same MoE sources also support the serialized Quark MXFP4 GLM-5.3
checkpoint: its routed and shared experts retain the GLM-5.2 ABI (hidden size
6144, intermediate size 2048, 256 routed experts, top-8 routing, one shared
expert, and identical packed-weight/scale tensor metadata). Integrations must
select a GLM-5.3 MoE-only profile so unrelated GLM-5.2 attention kernels are
not installed. The GLM-5.3 checkpoint may retain its native FP8 attention, but
FP8 expert weights and BF16 expert layouts other than the exact GLM NextN draft
ABI remain unsupported by this MoE bundle.

Other MoE families are not shape-compatible with the GLM snapshot. Qwen3-Next,
Kimi-Linear, and Mixtral still require model-specific kernels, profiles, weight
preparation, and contract tests.

DeepSeek-V4 Pro has a separate built-in TP8/EP1 gfx950 backend and kernel. It
accepts only the checkpoint's serialized FP4 routed experts with UE8M0 scales,
keeps the FP8 shared expert on its native linear path, and implements its
384-expert ungrouped sqrtsoftplus top-6 router and clamped SwiGLU. The first
three hash-routed layers remain on the native AMD runner. The Gluon kernel is
intentionally limited to the c=1 decode shapes M=1 (target) and M=4/6 (MTP);
other active-token counts, DeepSeek-V4 variants, FP8/BF16 experts, TP/EP
layouts other than TP8/EP1, and non-gfx950 devices fail explicitly.
Re-quantizing those experts as serialized Quark W4A4 MXFP4 is not sufficient:
the GLM kernels still require hidden size 6144, MoE intermediate size 2048, 256
routed experts, top-8 normalized sigmoid routing, and one shared expert. The
positive routed scaling factor is supplied by the model at runtime; model depth,
the first MoE layer, and MoE layer frequency are not kernel ABI constraints.
