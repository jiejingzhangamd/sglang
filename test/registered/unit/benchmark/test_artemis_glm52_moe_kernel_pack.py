from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOT = Path(__file__).parents[4]
PACK = ROOT / "benchmark" / "kernels" / "fused_moe_triton" / "artemis_glm52_tp8"


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
    assert len(referenced_sources) == 18
    assert shipped_sources == referenced_sources

    for row in specializations:
        source = PACK / row["source_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
