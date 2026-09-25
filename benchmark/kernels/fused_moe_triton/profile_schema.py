"""Load compact fused-MoE kernel profiles.

Schema v3 groups cases by source file and stores each case as:

    [candidate_id, signature, signature_ranges?, semantics?]

The optional semantics value overrides the profile-level default.  Consumers
receive the same flat ``specializations`` rows used by schema v2.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def expand_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Expand a schema-v3 profile while preserving schema-v2 compatibility."""
    schema_version = profile.get("schema_version")
    if schema_version == 2:
        return profile
    if schema_version != 3:
        raise ValueError(f"unsupported profile schema version: {schema_version}")

    variant = profile["variant"]
    default_semantics = profile.get("default_semantics", {})
    specializations = []

    for family in profile["families"]:
        source_file = family["source_file"]
        source_sha256 = family["source_sha256"]
        for case in family["cases"]:
            if not 2 <= len(case) <= 4:
                raise ValueError(f"invalid compact profile case: {case!r}")

            candidate_id, signature = case[:2]
            signature_ranges = case[2] if len(case) >= 3 else None
            semantics = case[3] if len(case) == 4 else default_semantics
            row = {
                "variant": variant,
                "candidate_id": candidate_id,
                "signature": signature,
                "source_file": source_file,
                "source_sha256": source_sha256,
                "semantics": deepcopy(semantics),
            }
            if signature_ranges is not None:
                row["signature_ranges"] = signature_ranges
            specializations.append(row)

    return {
        "schema_version": 2,
        "variant": variant,
        "specializations": specializations,
    }


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load and normalize a schema-v2 or compact schema-v3 profile."""
    return expand_profile(json.loads(Path(path).read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    json.dump(load_profile(args.profile), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
