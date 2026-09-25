#!/usr/bin/env python3
"""Capture and compare paired SGLang server traces for a MoE-only A/B test.

The tool deliberately keeps server launch outside the benchmark. This makes it
possible to compare a stock server with any plugin or source-level candidate
while holding the model, launch arguments, prompt, and profiling window fixed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

DEFAULT_PROMPT = (
    "Explain why fused mixture-of-experts kernels matter for low-concurrency decode."
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        return {"text": response.text}


def _generate(
    url: str,
    prompt: str,
    output_len: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
    }
    return request, _post_json(f"{url.rstrip('/')}/generate", request, timeout)


def capture(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_request, warmup_response = _generate(
        args.url, args.prompt, args.warmup_output_len, args.timeout
    )
    _write_json(output_dir / "warmup_request.json", warmup_request)
    _write_json(output_dir / "warmup_response.json", warmup_response)

    with ThreadPoolExecutor(max_workers=1) as executor:
        measurement = executor.submit(
            _generate,
            args.url,
            args.prompt,
            args.output_len,
            args.timeout,
        )
        time.sleep(args.profile_delay)

        server_output_dir = args.server_output_dir or str(output_dir)
        profile_request = {
            "output_dir": server_output_dir,
            "num_steps": args.steps,
            "activities": args.activities,
            "with_stack": False,
            "record_shapes": False,
            "profile_id": args.profile_id,
            "detailed_annotations": True,
        }
        _write_json(output_dir / "profile_request.json", profile_request)
        profile_response = _post_json(
            f"{args.url.rstrip('/')}/start_profile",
            profile_request,
            args.timeout,
        )
        _write_json(output_dir / "profile_response.json", profile_response)
        measurement_request, measurement_response = measurement.result()

    _write_json(output_dir / "measurement_request.json", measurement_request)
    _write_json(output_dir / "measurement_response.json", measurement_response)
    _write_json(
        output_dir / "capture_manifest.json",
        {
            "url": args.url,
            "profile_id": args.profile_id,
            "steps": args.steps,
            "profile_delay_seconds": args.profile_delay,
            "warmup_output_len": args.warmup_output_len,
            "measurement_output_len": args.output_len,
            "activities": args.activities,
        },
    )
    print(output_dir)


def _load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _gpu_events(path: Path) -> list[dict[str, Any]]:
    result = []
    for event in _load_trace(path).get("traceEvents", []):
        if event.get("ph") != "X" or not event.get("dur"):
            continue
        category = str(event.get("cat", "")).lower()
        event_args = event.get("args") or {}
        has_device_stream = (
            event_args.get("device") is not None
            and event_args.get("stream") is not None
        )
        if "kernel" in category or "gpu_" in category or has_device_stream:
            result.append(event)
    if not result:
        raise RuntimeError(f"No GPU activity events found in {path}")
    return result


def _interval_union_us(events: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in events
    )
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _find_rank_trace(directory: Path, rank: int) -> Path:
    patterns = (
        f"*TP-{rank}*.trace.json.gz",
        f"*TP-{rank}*.trace.json",
    )
    matches = sorted(path for pattern in patterns for path in directory.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one TP-{rank} trace below {directory}, found {matches}"
        )
    return matches[0]


def _extract_text(response_path: Path) -> str | list[str] | None:
    response = json.loads(response_path.read_text())
    text = response.get("text")
    if text is None and isinstance(response.get("data"), dict):
        text = response["data"].get("text")
    return text


def _text_digest(text: str | list[str] | None) -> str | None:
    if text is None:
        return None
    encoded = json.dumps(text, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _summarize_arm(directory: Path, rank: int, steps: int) -> dict[str, Any]:
    trace_path = _find_rank_trace(directory, rank)
    events = _gpu_events(trace_path)
    sum_us = sum(float(event["dur"]) for event in events)
    union_us = _interval_union_us(events)
    by_name: Counter[str] = Counter()
    for event in events:
        by_name[str(event.get("name", "<unnamed>"))] += float(event["dur"])

    response_path = directory / "measurement_response.json"
    text = _extract_text(response_path) if response_path.exists() else None
    return {
        "trace": str(trace_path),
        "rank": rank,
        "steps": steps,
        "gpu_activity_events": len(events),
        "gpu_sum_ms": sum_us / 1000.0,
        "gpu_union_ms": union_us / 1000.0,
        "gpu_sum_ms_per_step": sum_us / 1000.0 / steps,
        "gpu_union_ms_per_step": union_us / 1000.0 / steps,
        "output_sha256": _text_digest(text),
        "top_gpu_activities_by_sum_ms": [
            {"name": name, "sum_ms": duration / 1000.0}
            for name, duration in by_name.most_common(30)
        ],
    }


def compare(args: argparse.Namespace) -> None:
    baseline = _summarize_arm(args.baseline_dir, args.rank, args.steps)
    candidate = _summarize_arm(args.candidate_dir, args.rank, args.steps)
    delta_sum = baseline["gpu_sum_ms_per_step"] - candidate["gpu_sum_ms_per_step"]
    delta_union = baseline["gpu_union_ms_per_step"] - candidate["gpu_union_ms_per_step"]
    delta_sum_per_token = delta_sum / args.accepted_tokens_per_step
    delta_union_per_token = delta_union / args.accepted_tokens_per_step

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_minus_candidate": {
            "gpu_sum_ms_per_step": delta_sum,
            "gpu_union_ms_per_step": delta_union,
            "gpu_sum_ms_per_output_token": delta_sum_per_token,
            "gpu_union_ms_per_output_token": delta_union_per_token,
            "accepted_tokens_per_step": args.accepted_tokens_per_step,
        },
        "outputs_match": (
            baseline["output_sha256"] is not None
            and baseline["output_sha256"] == candidate["output_sha256"]
        ),
    }
    if args.target_ms_per_output_token is not None:
        report["target"] = {
            "ms_per_output_token": args.target_ms_per_output_token,
            "union_delta_minus_target_ms": (
                delta_union_per_token - args.target_ms_per_output_token
            ),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser(
        "capture", help="Capture a fixed decode profile from a running server"
    )
    capture_parser.add_argument("--url", default="http://127.0.0.1:30000")
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument(
        "--server-output-dir",
        help="Path visible to the server; defaults to the resolved output directory",
    )
    capture_parser.add_argument("--profile-id", required=True)
    capture_parser.add_argument("--steps", type=int, default=50)
    capture_parser.add_argument("--profile-delay", type=float, default=2.0)
    capture_parser.add_argument("--warmup-output-len", type=int, default=128)
    capture_parser.add_argument("--output-len", type=int, default=2048)
    capture_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    capture_parser.add_argument("--activities", nargs="+", default=["CPU", "GPU"])
    capture_parser.add_argument("--timeout", type=float, default=1800.0)
    capture_parser.set_defaults(func=capture)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare rank-local GPU activity from two captured profiles"
    )
    compare_parser.add_argument("--baseline-dir", type=Path, required=True)
    compare_parser.add_argument("--candidate-dir", type=Path, required=True)
    compare_parser.add_argument("--rank", type=int, default=0)
    compare_parser.add_argument("--steps", type=int, default=50)
    compare_parser.add_argument("--accepted-tokens-per-step", type=float, default=1.0)
    compare_parser.add_argument("--target-ms-per-output-token", type=float)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if getattr(args, "accepted_tokens_per_step", 1.0) <= 0:
        raise ValueError("--accepted-tokens-per-step must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
