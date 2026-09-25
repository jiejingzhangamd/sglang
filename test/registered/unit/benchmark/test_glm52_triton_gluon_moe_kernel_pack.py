from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOT = Path(__file__).parents[4]
PACK_ROOT = ROOT / "benchmark" / "kernels" / "fused_moe_triton"
TP4_PACK = PACK_ROOT / "glm52_triton_gluon_tp4"
TP8_PACK = PACK_ROOT / "glm52_triton_gluon_tp8"


def _assert_hash_verified_pack(pack: Path, rows: int, sources: int) -> dict:
    profile = json.loads((pack / "profile.json").read_text())
    specializations = profile["specializations"]

    assert profile["variant"] == "glm52.fused_moe"
    assert len(specializations) == rows
    assert {row["signature"][1] for row in specializations} == {0, 1}

    referenced_sources = {row["source_file"] for row in specializations}
    shipped_sources = {
        path.relative_to(pack).as_posix() for path in (pack / "fused_moe").glob("*.py")
    }
    assert len(referenced_sources) == sources
    assert shipped_sources == referenced_sources

    for row in specializations:
        source = pack / row["source_file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]

    return profile


def test_profile_references_complete_hash_verified_kernel_snapshot() -> None:
    _assert_hash_verified_pack(TP4_PACK, rows=42, sources=4)
    _assert_hash_verified_pack(TP8_PACK, rows=48, sources=8)


def test_related_shapes_share_compile_time_specialized_sources() -> None:
    profile = json.loads((TP8_PACK / "profile.json").read_text())
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
    assert {sources[(shape, 1)] for shape in (1, 2, 4, 6, 8, 12, 16)} == {
        "fused_moe_tp8_m1_16_mtp.py"
    }
    assert {
        sources[(shape, 1)]
        for shape in (24, 32, 48, 64, 96, 128, 192, 256, 384, 768)
    } == {
        "fused_moe_tp8_m32_256_mtp.py"
    }


def test_tp4_related_shapes_share_compile_time_specialized_sources() -> None:
    profile = json.loads((TP4_PACK / "profile.json").read_text())
    sources = {
        tuple(row["signature"][:2]): Path(row["source_file"]).name
        for row in profile["specializations"]
        if not row.get("signature_ranges")
    }

    for mode in (0, 1):
        assert {sources[(shape, mode)] for shape in (1, 2, 4, 8, 16)} == {
            "fused_moe_tp4_m1_16.py"
        }
        assert {sources[(shape, mode)] for shape in (32, 64)} == {
            "fused_moe_tp4_m32_64.py"
        }
        medium_shapes = (128, 256, 1024, 2048, 3072, 4096, 4192)
        assert {sources[(shape, mode)] for shape in medium_shapes} == {
            "fused_moe_tp4_m128_4192.py"
        }
