"""Evaluate prompt-faithfulness canaries for the LEGOGen brick pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.brick.parser import parse_brick_sequence
from scripts.benchmark_legogen import (
    DEFAULT_OUTPUT_ROOT,
    _benchmark_limits,
    _elapsed_ms,
    _has_generation_limits,
    _limit_row_fields,
    _markdown_table,
    _now_timestamp,
    _rate,
    _safe_mean,
    _write_csv,
    _write_json,
    _write_jsonl,
    collect_environment_metadata,
    real_skip_reason,
    validate_bricks_text,
)

DEFAULT_QUALITY_PROMPTS = PROJECT_ROOT / "benchmarks" / "prompts" / "quality_canary_prompts.jsonl"


def read_quality_prompts(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if "prompt" not in row:
            raise ValueError(f"{path}:{line_no} is missing required 'prompt'")
        row.setdefault("id", f"prompt-{line_no}")
        row.setdefault("expected_colors", [])
        row.setdefault("min_distinct_colors", len(row["expected_colors"]))
        row.setdefault("min_bricks", 1)
        prompts.append(row)
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No quality prompts found in {path}")
    return prompts


def _quality_checks(bricks_text: str, expected_colors: list[str], min_distinct_colors: int, min_bricks: int) -> dict[str, Any]:
    try:
        bricks = parse_brick_sequence(bricks_text)
    except Exception:
        bricks = []
    colors = sorted({brick.color for brick in bricks})
    expected = [color.upper().lstrip("#") for color in expected_colors]
    found = [color for color in expected if color in colors]
    missing = [color for color in expected if color not in colors]
    coverage = round(len(found) / len(expected), 6) if expected else 1.0
    return {
        "distinct_colors": colors,
        "distinct_color_count": len(colors),
        "expected_colors": expected,
        "expected_colors_found": found,
        "expected_colors_missing": missing,
        "expected_color_coverage": coverage,
        "expected_colors_all_present": not missing,
        "min_distinct_colors": min_distinct_colors,
        "meets_min_distinct_colors": len(colors) >= min_distinct_colors,
        "min_bricks": min_bricks,
        "meets_min_bricks": len(bricks) >= min_bricks,
    }


def run_quality_eval(
    *,
    prompts_path: Path = DEFAULT_QUALITY_PROMPTS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    limit_prompts: int | None = None,
    max_bricks_per_sample: int | None = None,
    sample_timeout_s: float | None = None,
    stability_check_interval: int | None = None,
    allow_base_model: bool = False,
    pipeline: Any | None = None,
) -> Path:
    timestamp = timestamp or _now_timestamp()
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    limits = _benchmark_limits(
        quick_smoke=False,
        max_bricks_per_sample=max_bricks_per_sample,
        sample_timeout_s=sample_timeout_s,
        stability_check_interval=stability_check_interval,
    )
    prompts = read_quality_prompts(prompts_path, limit=limit_prompts)
    config = {
        "prompts_path": str(prompts_path),
        "limit_prompts": limit_prompts,
        "allow_base_model": allow_base_model,
        **limits,
    }
    _write_json(run_dir / "quality_config.json", config)
    _write_json(run_dir / "environment_metadata.json", collect_environment_metadata(limits))
    metadata = json.loads((run_dir / "environment_metadata.json").read_text(encoding="utf-8"))
    run_mode = metadata["environment"]["run_mode"]

    skip_reason = real_skip_reason(metadata, allow_base_model=allow_base_model)
    if skip_reason is not None:
        rows: list[dict[str, Any]] = []
        summary = summarize_quality(rows, run_mode=run_mode, limits=limits)
        _write_jsonl(run_dir / "quality_raw.jsonl", rows)
        _write_csv(run_dir / "quality_summary.csv", summary, list(summary[0].keys()))
        write_quality_summary_md(run_dir / "quality_summary.md", summary, skip_reason=skip_reason)
        write_quality_report(run_dir, metadata=metadata, summary=summary, prompt_count=len(prompts), skip_reason=skip_reason)
        return run_dir

    if pipeline is None:
        try:
            from backend.inference.brick_pipeline import get_brick_pipeline

            pipeline = get_brick_pipeline()
        except Exception as exc:
            skip_reason = f"Could not initialize brick pipeline: {exc!r}"
            rows = []
            summary = summarize_quality(rows, run_mode=run_mode, limits=limits)
            _write_jsonl(run_dir / "quality_raw.jsonl", rows)
            _write_csv(run_dir / "quality_summary.csv", summary, list(summary[0].keys()))
            write_quality_summary_md(run_dir / "quality_summary.md", summary, skip_reason=skip_reason)
            write_quality_report(run_dir, metadata=metadata, summary=summary, prompt_count=len(prompts), skip_reason=skip_reason)
            return run_dir

    rows: list[dict[str, Any]] = []
    for idx, prompt_row in enumerate(prompts, start=1):
        row: dict[str, Any] = {
            "mode": "quality",
            "run_mode": run_mode,
            "prompt_id": idx,
            "quality_id": prompt_row["id"],
            "prompt": prompt_row["prompt"],
            "success": False,
            **_limit_row_fields(limits),
        }
        start = time.perf_counter()
        try:
            if _has_generation_limits(limits):
                result = pipeline.generate(
                    prompt_row["prompt"],
                    max_bricks=limits["max_bricks_per_sample"],
                    max_seconds=limits["sample_timeout_s"],
                    stability_check_interval=limits["stability_check_interval"],
                )
            else:
                result = pipeline.generate(prompt_row["prompt"])
            metadata_row = result.get("metadata", {}) or {}
            bricks_text = result.get("bricks", "")
            row.update({
                "success": True,
                "wall_time_ms": _elapsed_ms(start),
                "brick_count": result.get("brick_count"),
                "stable": result.get("stable"),
                "termination_reason": metadata_row.get("termination_reason"),
                "hit_done": metadata_row.get("hit_done"),
                "hit_max_bricks": metadata_row.get("hit_max_bricks"),
                "hit_max_seconds": metadata_row.get("hit_max_seconds"),
                "rollbacks": metadata_row.get("rollbacks"),
                "rejections": metadata_row.get("rejections"),
                "metadata": metadata_row,
                "bricks": bricks_text,
                **validate_bricks_text(bricks_text, title=f"quality-{prompt_row['id']}"),
                **_quality_checks(
                    bricks_text,
                    prompt_row.get("expected_colors", []),
                    int(prompt_row.get("min_distinct_colors", 0)),
                    int(prompt_row.get("min_bricks", 1)),
                ),
            })
        except Exception as exc:
            row.update({
                "wall_time_ms": _elapsed_ms(start),
                "error": repr(exc),
            })
        rows.append(row)

    summary = summarize_quality(rows, run_mode=run_mode, limits=limits)
    _write_jsonl(run_dir / "quality_raw.jsonl", rows)
    _write_csv(run_dir / "quality_summary.csv", summary, list(summary[0].keys()))
    write_quality_summary_md(run_dir / "quality_summary.md", summary)
    write_quality_report(run_dir, metadata=metadata, summary=summary, prompt_count=len(prompts))
    return run_dir


def summarize_quality(rows: list[dict[str, Any]], *, run_mode: str, limits: dict[str, Any]) -> list[dict[str, Any]]:
    successes = [row for row in rows if row.get("success")]
    return [{
        "run_mode": run_mode,
        **_limit_row_fields(limits),
        "prompt_count": len(rows),
        "success_count": len(successes),
        "success_rate": _rate(row.get("success") for row in rows),
        "parse_valid_rate": _rate(row.get("parse_valid") for row in successes),
        "collision_free_rate": _rate(row.get("collision_free") for row in successes),
        "stable_rate": _rate(row.get("stable") for row in successes),
        "export_success_rate": _rate(row.get("export_success") for row in successes),
        "expected_colors_all_present_rate": _rate(row.get("expected_colors_all_present") for row in successes),
        "avg_expected_color_coverage": _safe_mean(row.get("expected_color_coverage") for row in successes),
        "min_distinct_colors_rate": _rate(row.get("meets_min_distinct_colors") for row in successes),
        "min_bricks_rate": _rate(row.get("meets_min_bricks") for row in successes),
        "done_stop_rate": _rate(row.get("hit_done") for row in successes),
        "hit_max_bricks_rate": _rate(row.get("hit_max_bricks") for row in successes),
        "avg_brick_count": _safe_mean(row.get("brick_count") for row in successes),
        "avg_wall_time_ms": _safe_mean(row.get("wall_time_ms") for row in successes),
    }]


def write_quality_summary_md(path: Path, summary: list[dict[str, Any]], *, skip_reason: str | None = None) -> None:
    note = "Prompt-faithfulness canary summary for the current brick-coordinate pipeline."
    if skip_reason is not None:
        note += f"\n\nGeneration skipped: {skip_reason}"
    headers = list(summary[0].keys()) if summary else ["run_mode"]
    path.write_text("# LEGOGen Quality Canary Summary\n\n" + note + "\n\n" + _markdown_table(summary, headers), encoding="utf-8")


def write_quality_report(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    summary: list[dict[str, Any]],
    prompt_count: int,
    skip_reason: str | None = None,
) -> None:
    lines = [
        "# LEGOGen Quality Canary Report",
        "",
        f"- Run mode: `{metadata['environment']['run_mode']}`",
        f"- LEGOGEN_DEV: `{metadata['environment']['LEGOGEN_DEV_raw']}`",
        f"- Prompts: `{prompt_count}`",
        f"- Git commit: `{metadata['git']['commit'] or 'unknown'}`",
        "- Purpose: check prompt color coverage, natural DONE stopping, validity, and exportability.",
    ]
    if metadata["environment"]["run_mode"] == "dev-mock":
        lines.append("- Note: dev/mock results are smoke checks only, not model quality.")
    if skip_reason is not None:
        lines.extend(["", f"Generation skipped: {skip_reason}"])
    if summary:
        lines.extend(["", "## Summary", ""])
        for key, value in summary[0].items():
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    (run_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LEGOGen quality canary evaluation.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_QUALITY_PROMPTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--allow-base-model", action="store_true")
    parser.add_argument("--max-bricks-per-sample", type=_positive_int, default=None)
    parser.add_argument("--sample-timeout-s", type=_positive_float, default=None)
    parser.add_argument("--stability-check-interval", type=_positive_int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_dir = run_quality_eval(
        prompts_path=args.prompts,
        output_root=args.output_root,
        timestamp=args.timestamp,
        limit_prompts=args.limit_prompts,
        allow_base_model=args.allow_base_model,
        max_bricks_per_sample=args.max_bricks_per_sample,
        sample_timeout_s=args.sample_timeout_s,
        stability_check_interval=args.stability_check_interval,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
