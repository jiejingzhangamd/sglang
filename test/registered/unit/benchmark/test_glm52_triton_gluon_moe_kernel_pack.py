from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmark.kernels.fused_moe_triton.profile_schema import (
    expand_profile,
    load_profile,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOT = Path(__file__).parents[4]
PACK_ROOT = ROOT / "benchmark" / "kernels" / "fused_moe_triton"
TP4_PACK = PACK_ROOT / "glm52_triton_gluon_tp4"
TP8_PACK = PACK_ROOT / "glm52_triton_gluon_tp8"


def test_compact_profile_expands_to_schema_v2_rows() -> None:
    compact = {
        "schema_version": 3,
        "variant": "test.variant",
        "default_semantics": {"contract": {"family": "moe"}},
        "families": [
            {
                "source_file": "kernel.py",
                "source_sha256": "digest",
                "cases": [
                    ["exact", [1, 0, 0]],
                    ["range", [2, 1, 0], {"0": [2, 4]}, {"contract": {}}],
                ],
            }
        ],
    }

    expanded = expand_profile(compact)
    exact, ranged = expanded["specializations"]
    assert expanded["schema_version"] == 2
    assert exact["semantics"] == {"contract": {"family": "moe"}}
    assert "signature_ranges" not in exact
    assert ranged["signature_ranges"] == {"0": [2, 4]}
    assert ranged["semantics"] == {"contract": {}}
    assert expand_profile(expanded) is expanded


def _load_profile(pack: Path) -> dict:
    return load_profile(pack / "profile.json")


def _exact_source_map(pack: Path) -> dict[tuple[int, int], str]:
    return {
        tuple(row["signature"][:2]): Path(row["source_file"]).name
        for row in _load_profile(pack)["specializations"]
        if not row.get("signature_ranges")
    }


def _resolve_source(profile: dict, signature: tuple[int, ...]) -> str:
    rows = profile["specializations"]
    exact_matches = [
        row
        for row in rows
        if not row.get("signature_ranges") and tuple(row["signature"]) == signature
    ]
    if exact_matches:
        assert len(exact_matches) == 1
        return Path(exact_matches[0]["source_file"]).name

    range_matches = []
    for row in rows:
        ranges = row.get("signature_ranges")
        if not ranges or len(row["signature"]) != len(signature):
            continue

        matches = True
        for index, actual in enumerate(signature):
            bounds = ranges.get(str(index))
            if bounds is None:
                matches = matches and actual == row["signature"][index]
            else:
                matches = matches and bounds[0] <= actual <= bounds[1]
        if matches:
            range_matches.append(row)

    assert len(range_matches) == 1, (
        f"expected one source for signature {signature}, got {len(range_matches)}"
    )
    return Path(range_matches[0]["source_file"]).name


def _assert_hash_verified_pack(pack: Path, rows: int, sources: int) -> dict:
    compact_profile = json.loads((pack / "profile.json").read_text())
    assert compact_profile["schema_version"] == 3
    assert "specializations" not in compact_profile
    assert len(compact_profile["families"]) == sources
    assert sum(len(family["cases"]) for family in compact_profile["families"]) == rows

    profile = _load_profile(pack)
    specializations = profile["specializations"]

    assert profile["schema_version"] == 2
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
    sources = _exact_source_map(TP8_PACK)
    families = (
        ((1, 2, 4, 8, 16), 0, "fused_moe_tp8_m1_16.py"),
        ((32, 64, 128), 0, "fused_moe_tp8_m32_128.py"),
        ((1, 2, 4, 6, 8, 12, 16), 1, "fused_moe_tp8_m1_16_mtp.py"),
        (
            (24, 32, 48, 64, 96, 128, 192, 256, 384, 768),
            1,
            "fused_moe_tp8_m32_256_mtp.py",
        ),
    )

    for shapes, mode, expected_source in families:
        assert {sources[(shape, mode)] for shape in shapes} == {expected_source}


def test_tp8_mtp_product_shapes_resolve_to_large_mtp_source() -> None:
    profile = _load_profile(TP8_PACK)
    expected_source = "fused_moe_tp8_m1024_4192_mtp.py"

    for batch_size in (341, 561):
        for draft_width in (4, 6):
            signature = (batch_size * draft_width, 1, 0)
            assert _resolve_source(profile, signature) == expected_source


def test_tp4_related_shapes_share_compile_time_specialized_sources() -> None:
    sources = _exact_source_map(TP4_PACK)

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
