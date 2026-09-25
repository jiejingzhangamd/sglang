from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOT = Path(__file__).parents[4]
SCRIPT = (
    ROOT / "benchmark" / "kernels" / "fused_moe_triton" / "profile_moe_server_ab.py"
)
SPEC = importlib.util.spec_from_file_location("profile_moe_server_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_trace(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def test_interval_union_handles_overlapping_gpu_activity() -> None:
    events = [
        {"ts": 10, "dur": 5},
        {"ts": 12, "dur": 10},
        {"ts": 30, "dur": 2},
    ]
    assert MODULE._interval_union_us(events) == 14


def test_summarize_arm_filters_cpu_events_and_hashes_output(tmp_path: Path) -> None:
    trace = tmp_path / "profile-TP-0.trace.json.gz"
    _write_trace(
        trace,
        [
            {"ph": "X", "cat": "cpu_op", "name": "aten::empty", "ts": 1, "dur": 50},
            {"ph": "X", "cat": "kernel", "name": "moe_a", "ts": 10, "dur": 20},
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "copy",
                "ts": 25,
                "dur": 10,
            },
        ],
    )
    (tmp_path / "measurement_response.json").write_text(
        json.dumps({"text": "deterministic output"})
    )

    summary = MODULE._summarize_arm(tmp_path, rank=0, steps=5)

    assert summary["gpu_activity_events"] == 2
    assert summary["gpu_sum_ms_per_step"] == 0.006
    assert summary["gpu_union_ms_per_step"] == 0.005
    assert summary["output_sha256"] is not None


def test_find_rank_trace_rejects_ambiguous_profiles(tmp_path: Path) -> None:
    for name in ("a-TP-0.trace.json.gz", "b-TP-0.trace.json.gz"):
        _write_trace(tmp_path / name, [])

    try:
        MODULE._find_rank_trace(tmp_path, rank=0)
    except RuntimeError as error:
        assert "Expected one TP-0 trace" in str(error)
    else:
        raise AssertionError("ambiguous rank traces must be rejected")
