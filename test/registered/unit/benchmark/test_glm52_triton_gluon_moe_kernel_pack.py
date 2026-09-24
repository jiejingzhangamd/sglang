from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOT = Path(__file__).parents[4]
PACK = ROOT / "benchmark" / "kernels" / "fused_moe_triton" / "glm52_triton_gluon_tp8"


def test_profile_references_complete_hash_verified_kernel_snapshot() -> None:
    profile = json.loads((PACK / "profile.json").read_text())
    specializations = profile["specializations"]

    assert profile["variant"] == "glm52.fused_moe"
    assert len(specializations) == 40
    assert {row["signature"][1] for row in specializations} == {0, 1}

    referenced_sources = {row["source_file"] for row in specializations}
    shipped_sources = {
        path.relative_to(PACK).as_posix() for path in (PACK / "fused_moe").glob("*.py")
    }
    assert len(referenced_sources) == 8
    assert shipped_sources == referenced_sources

    for row in specializations:
        source = PACK / row["source_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]


def test_related_shapes_share_compile_time_specialized_sources() -> None:
    profile = json.loads((PACK / "profile.json").read_text())
    sources = {
        tuple(row["signature"][:2]): Path(row["source_file"]).name
        for row in profile["specializations"]
        if not row.get("signature_ranges")
    }

    assert {sources[(shape, 0)] for shape in (1, 2, 4, 8, 16)} == {
        "fused_moe_tp8_m1_16.py"
    }
    assert {sources[(shape, 0)] for shape in (32, 64, 128)} == {
        "fused_moe_tp8_m32_128.py"
    }
    assert {sources[(shape, 1)] for shape in (1, 2, 4, 8, 16)} == {
        "fused_moe_tp8_m1_16_mtp.py"
    }
    assert {sources[(shape, 1)] for shape in (32, 64, 128, 256)} == {
        "fused_moe_tp8_m32_256_mtp.py"
    }
