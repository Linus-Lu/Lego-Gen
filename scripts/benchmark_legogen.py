"""Benchmark the current LEGOGen brick-coordinate pipeline.

This harness measures the code paths that exist today:
  - n=1 text generation through BrickPipeline.generate()
  - Best-of-N through BrickPipeline.generate_best_of_n()
  - route-level require_stable behavior through FastAPI TestClient
  - LDraw export through backend.brick.ldraw.export_ldr()

It deliberately does not benchmark removed JSON or validator endpoints.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.brick.ldraw import export_ldr
from backend.brick.occupancy import VoxelGrid
from backend.brick.parser import Brick, parse_brick_sequence
from backend.brick.stability import is_stable
from backend.config import (
    BRICK_CHECKPOINT_DIR,
    BRICK_MODEL_NAME,
    LEGOGEN_DEV,
    STAGE1_CHECKPOINT_DIR,
    STAGE1_MODEL_NAME,
)

DEFAULT_NS = [1, 2, 4, 8, 16]
DEFAULT_PROMPTS = PROJECT_ROOT / "benchmarks" / "prompts" / "core_prompts.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_runs"
ALL_MODES = ("core", "bon", "stable-only", "export")
QUICK_SMOKE_MODES = ("core", "export")
QUICK_SMOKE_LIMIT_PROMPTS = 1
QUICK_SMOKE_MAX_BRICKS = 24
QUICK_SMOKE_SAMPLE_TIMEOUT_S = 90.0
QUICK_SMOKE_STABILITY_CHECK_INTERVAL = 8


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _safe_mean(values: Iterable[float | int | None]) -> float:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else 0.0


def _safe_p50(values: Iterable[float | int | None]) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    return round(statistics.median(vals), 3)


def _rate(values: Iterable[Any]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return round(sum(1 for v in vals if bool(v)) / len(vals), 6)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _benchmark_limits(
    *,
    quick_smoke: bool,
    max_bricks_per_sample: int | None,
    sample_timeout_s: float | None,
    stability_check_interval: int | None,
) -> dict[str, Any]:
    capped_generation = any(
        value is not None
        for value in (max_bricks_per_sample, sample_timeout_s, stability_check_interval)
    )
    return {
        "quick_smoke": quick_smoke,
        "capped_generation": capped_generation,
        "max_bricks_per_sample": max_bricks_per_sample,
        "sample_timeout_s": sample_timeout_s,
        "stability_check_interval": stability_check_interval,
    }


def _limit_row_fields(limits: dict[str, Any]) -> dict[str, Any]:
    return {
        "quick_smoke": limits["quick_smoke"],
        "capped_generation": limits["capped_generation"],
        "max_bricks_per_sample": limits["max_bricks_per_sample"],
        "sample_timeout_s": limits["sample_timeout_s"],
        "stability_check_interval": limits["stability_check_interval"],
    }


def _has_generation_limits(limits: dict[str, Any]) -> bool:
    return any(
        limits[key] is not None
        for key in ("max_bricks_per_sample", "sample_timeout_s", "stability_check_interval")
    )


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _write_summary_md(path: Path, title: str, note: str, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.write_text(f"# {title}\n\n{note}\n\n{_markdown_table(rows, headers)}", encoding="utf-8")


def _run_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_environment_metadata(benchmark_limits: dict[str, Any] | None = None) -> dict[str, Any]:
    torch_meta: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu_names = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ] if cuda_available else []
        torch_meta = {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "gpu_names": gpu_names,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
        }
    except Exception as exc:
        torch_meta = {
            "available": False,
            "error": repr(exc),
            "cuda_available": False,
            "device_count": 0,
            "gpu_names": [],
        }

    status_short = _run_git(["status", "--short"])
    metadata = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "status_short": status_short,
            "dirty": bool(status_short),
        },
        "environment": {
            "LEGOGEN_DEV_raw": os.environ.get("LEGOGEN_DEV"),
            "LEGOGEN_DEV_effective": bool(LEGOGEN_DEV),
            "run_mode": "dev-mock" if LEGOGEN_DEV else "real",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "models": {
            "stage1_model_id": STAGE1_MODEL_NAME,
            "brick_model_id": BRICK_MODEL_NAME,
            "stage1_checkpoint_dir": str(STAGE1_CHECKPOINT_DIR),
            "brick_checkpoint_dir": str(BRICK_CHECKPOINT_DIR),
            "stage1_checkpoint_present": STAGE1_CHECKPOINT_DIR.exists(),
            "brick_checkpoint_present": BRICK_CHECKPOINT_DIR.exists(),
            "stage1_adapter_config_present": (STAGE1_CHECKPOINT_DIR / "adapter_config.json").exists(),
            "brick_adapter_config_present": (BRICK_CHECKPOINT_DIR / "adapter_config.json").exists(),
        },
        "torch": torch_meta,
        "outlines": {
            "available": importlib.util.find_spec("outlines") is not None,
        },
    }
    if benchmark_limits is not None:
        metadata["benchmark_limits"] = benchmark_limits
    return metadata


def default_comparison_label() -> str:
    adapter_path = BRICK_CHECKPOINT_DIR / "adapter_config.json"
    if adapter_path.exists():
        return BRICK_CHECKPOINT_DIR.name
    return f"base::{BRICK_MODEL_NAME}"


def _normalize_device_identifier(device: Any) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def _find_hf_device_map(model: Any) -> dict[str, str] | None:
    if model is None:
        return None
    seen: set[int] = set()
    queue = [model]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        device_map = getattr(current, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            return {
                str(module_name): _normalize_device_identifier(device)
                for module_name, device in device_map.items()
            }
        for attr in ("model", "base_model", "module", "model_wrapped"):
            child = getattr(current, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)
    return None


def _collect_model_devices(model: Any, hf_device_map: dict[str, str] | None) -> list[str]:
    devices = set(hf_device_map.values()) if hf_device_map else set()
    if model is not None and not devices:
        try:
            for param in model.parameters():
                devices.add(str(param.device))
                if len(devices) >= 8:
                    break
        except Exception:
            pass
    return sorted(devices)


def _supports_generation_limits(method: Any) -> bool:
    if method is None:
        return False
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    required = {"max_bricks", "max_seconds", "stability_check_interval"}
    return required.issubset(params.keys())


def collect_pipeline_metadata(pipeline: Any, *, comparison_label: str) -> dict[str, Any]:
    model = getattr(pipeline, "model", None)
    hf_device_map = _find_hf_device_map(model)
    model_devices = _collect_model_devices(model, hf_device_map)
    cuda_devices = sorted(device for device in model_devices if device.startswith("cuda"))
    return {
        "comparison_label": comparison_label,
        "pipeline_class": type(pipeline).__name__,
        "model_class": type(model).__name__ if model is not None else None,
        "supports_generate_best_of_n": callable(getattr(pipeline, "generate_best_of_n", None)),
        "supports_generation_limits": _supports_generation_limits(getattr(pipeline, "generate", None)),
        "hf_device_map": hf_device_map,
        "hf_device_map_devices": sorted(set(hf_device_map.values())) if hf_device_map else [],
        "hf_device_map_device_count": len(set(hf_device_map.values())) if hf_device_map else 0,
        "model_devices": model_devices,
        "cuda_model_devices": cuda_devices,
        "model_parallel_active": len(cuda_devices) > 1,
    }


def real_skip_reason(metadata: dict[str, Any], *, allow_base_model: bool = False) -> str | None:
    if metadata["environment"]["LEGOGEN_DEV_effective"]:
        return None
    if not metadata["torch"].get("cuda_available"):
        return "LEGOGEN_DEV=0 requested, but CUDA is unavailable."
    if not allow_base_model and not metadata["models"].get("brick_adapter_config_present"):
        return (
            "LEGOGEN_DEV=0 requested, but the Stage 2 brick LoRA checkpoint "
            f"is missing at {BRICK_CHECKPOINT_DIR}/adapter_config.json."
        )
    return None


def read_prompts(path: Path, *, limit: int | None = None) -> list[str]:
    prompts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def inspect_export(bricks: list[Brick], *, title: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        ldraw_text = export_ldr(bricks, title=title)
        export_time_ms = _elapsed_ms(start)
        lines = ldraw_text.splitlines()
        part_line_count = sum(1 for line in lines if line.startswith("1 "))
        header_present = (
            any(line.startswith("0 FILE ") for line in lines)
            and "0 Author: LEGOGen" in lines
            and "0 !LDRAW_ORG Model" in lines
        )
        return {
            "export_success": True,
            "export_error": None,
            "export_time_ms": export_time_ms,
            "ldraw_header_present": header_present,
            "ldraw_part_line_count": part_line_count,
            "ldraw_part_line_count_matches_bricks": part_line_count == len(bricks),
            "ldraw_text": ldraw_text,
        }
    except Exception as exc:
        return {
            "export_success": False,
            "export_error": repr(exc),
            "export_time_ms": _elapsed_ms(start),
            "ldraw_header_present": False,
            "ldraw_part_line_count": 0,
            "ldraw_part_line_count_matches_bricks": False,
            "ldraw_text": "",
        }


def validate_bricks_text(bricks_text: str, *, title: str = "LEGOGen Benchmark") -> dict[str, Any]:
    parse_valid = False
    parse_error = None
    bricks: list[Brick] = []
    try:
        bricks = parse_brick_sequence(bricks_text)
        parse_valid = True
    except Exception as exc:
        parse_error = repr(exc)

    collision_free: bool | None = None
    first_collision_index: int | None = None
    recomputed_stable: bool | None = None
    stability_error = None

    if parse_valid:
        grid = VoxelGrid()
        collision_free = True
        for idx, brick in enumerate(bricks):
            if not grid.can_place(brick):
                collision_free = False
                first_collision_index = idx
                break
            grid.place(brick)
        try:
            recomputed_stable = bool(is_stable(bricks))
        except Exception as exc:
            stability_error = repr(exc)

    export_info = inspect_export(bricks, title=title) if parse_valid else {
        "export_success": False,
        "export_error": "parse_invalid",
        "export_time_ms": 0,
        "ldraw_header_present": False,
        "ldraw_part_line_count": 0,
        "ldraw_part_line_count_matches_bricks": False,
        "ldraw_text": "",
    }
    export_info_no_text = {k: v for k, v in export_info.items() if k != "ldraw_text"}
    return {
        "parse_valid": parse_valid,
        "parse_error": parse_error,
        "recomputed_brick_count": len(bricks) if parse_valid else 0,
        "collision_free": collision_free,
        "first_collision_index": first_collision_index,
        "recomputed_stable": recomputed_stable,
        "stability_error": stability_error,
        **export_info_no_text,
    }


def run_core_benchmark(
    pipeline: Any,
    prompts: list[str],
    *,
    run_mode: str,
    limits: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    limits = limits or _benchmark_limits(
        quick_smoke=False,
        max_bricks_per_sample=None,
        sample_timeout_s=None,
        stability_check_interval=None,
    )
    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, start=1):
        row: dict[str, Any] = {
            "mode": "core",
            "run_mode": run_mode,
            "prompt_id": idx,
            "prompt": prompt,
            "success": False,
            **_limit_row_fields(limits),
        }
        start = time.perf_counter()
        try:
            if _has_generation_limits(limits):
                result = pipeline.generate(
                    prompt,
                    max_bricks=limits["max_bricks_per_sample"],
                    max_seconds=limits["sample_timeout_s"],
                    stability_check_interval=limits["stability_check_interval"],
                )
            else:
                result = pipeline.generate(prompt)
            wall_time_ms = _elapsed_ms(start)
            metadata = result.get("metadata", {}) or {}
            checks = validate_bricks_text(result.get("bricks", ""), title=f"core-{idx}")
            row.update({
                "success": True,
                "wall_time_ms": wall_time_ms,
                "generation_time_ms": metadata.get("generation_time_ms"),
                "brick_count": result.get("brick_count"),
                "stable": result.get("stable"),
                "final_stable": metadata.get("final_stable", result.get("stable")),
                "termination_reason": metadata.get("termination_reason"),
                "rejections": metadata.get("rejections"),
                "rollbacks": metadata.get("rollbacks"),
                "outlines_enabled": metadata.get("outlines_enabled"),
                "palette_validation_enabled": metadata.get("palette_validation_enabled"),
                "requested_max_bricks": metadata.get("requested_max_bricks"),
                "requested_max_seconds": metadata.get("requested_max_seconds"),
                "requested_stability_check_interval": metadata.get("requested_stability_check_interval"),
                "hit_max_bricks": metadata.get("hit_max_bricks"),
                "hit_max_seconds": metadata.get("hit_max_seconds"),
                "metadata": metadata,
                "bricks": result.get("bricks", ""),
                **checks,
            })
        except Exception as exc:
            row.update({
                "wall_time_ms": _elapsed_ms(start),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        rows.append(row)
    return rows


def run_bon_benchmark(
    pipeline: Any,
    prompts: list[str],
    *,
    ns: list[int],
    strategy: str,
    run_mode: str,
    limits: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    limits = limits or _benchmark_limits(
        quick_smoke=False,
        max_bricks_per_sample=None,
        sample_timeout_s=None,
        stability_check_interval=None,
    )
    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, start=1):
        for n in ns:
            row: dict[str, Any] = {
                "mode": "bon",
                "run_mode": run_mode,
                "prompt_id": idx,
                "prompt": prompt,
                "n": n,
                "strategy": strategy,
                "success": False,
                **_limit_row_fields(limits),
            }
            start = time.perf_counter()
            try:
                if _has_generation_limits(limits):
                    result = pipeline.generate_best_of_n(
                        prompt,
                        n=n,
                        strategy=strategy,
                        max_bricks=limits["max_bricks_per_sample"],
                        max_seconds=limits["sample_timeout_s"],
                        stability_check_interval=limits["stability_check_interval"],
                    )
                else:
                    result = pipeline.generate_best_of_n(prompt, n=n, strategy=strategy)
                wall_time_ms = _elapsed_ms(start)
                metadata = result.get("metadata", {}) or {}
                checks = validate_bricks_text(result.get("bricks", ""), title=f"bon-{idx}-n{n}")
                row.update({
                    "success": True,
                    "wall_time_ms": wall_time_ms,
                    "generation_time_ms": metadata.get("generation_time_ms"),
                    "brick_count": result.get("brick_count"),
                    "stable": result.get("stable"),
                    "final_stable": metadata.get("final_stable", result.get("stable")),
                    "stable_rate": metadata.get("stable_rate"),
                    "picked_index": metadata.get("picked_index"),
                    "termination_reason": metadata.get("termination_reason"),
                    "rejections": metadata.get("rejections"),
                    "rollbacks": metadata.get("rollbacks"),
                    "outlines_enabled": metadata.get("outlines_enabled"),
                    "palette_validation_enabled": metadata.get("palette_validation_enabled"),
                    "requested_max_bricks": metadata.get("requested_max_bricks"),
                    "requested_max_seconds": metadata.get("requested_max_seconds"),
                    "requested_stability_check_interval": metadata.get("requested_stability_check_interval"),
                    "hit_max_bricks": metadata.get("hit_max_bricks"),
                    "hit_max_seconds": metadata.get("hit_max_seconds"),
                    "metadata": metadata,
                    "bricks": result.get("bricks", ""),
                    **checks,
                })
            except Exception as exc:
                row.update({
                    "wall_time_ms": _elapsed_ms(start),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
            rows.append(row)
    return rows


def run_stable_only_benchmark(prompts: list[str], *, run_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from fastapi.testclient import TestClient
        from backend.app import app

        client = TestClient(app, raise_server_exceptions=False)
    except Exception as exc:
        return [
            {
                "mode": "stable-only",
                "run_mode": run_mode,
                "prompt_id": idx,
                "prompt": prompt,
                "success": False,
                "status_code": None,
                "error": f"Could not create TestClient: {exc!r}",
            }
            for idx, prompt in enumerate(prompts, start=1)
        ]

    try:
        for idx, prompt in enumerate(prompts, start=1):
            row: dict[str, Any] = {
                "mode": "stable-only",
                "run_mode": run_mode,
                "prompt_id": idx,
                "prompt": prompt,
                "route": "POST /api/generate-bricks",
                "require_stable": True,
                "success": False,
            }
            start = time.perf_counter()
            try:
                response = client.post(
                    "/api/generate-bricks",
                    data={"prompt": prompt, "require_stable": "true", "n": "1"},
                )
                row["wall_time_ms"] = _elapsed_ms(start)
                row["status_code"] = response.status_code
                try:
                    body = response.json()
                except Exception:
                    body = {"raw_text": response.text}
                row["response_body"] = body
                if response.status_code == 200:
                    metadata = body.get("metadata", {}) or {}
                    checks = validate_bricks_text(body.get("bricks", ""), title=f"stable-only-{idx}")
                    row.update({
                        "success": True,
                        "generation_time_ms": metadata.get("generation_time_ms"),
                        "brick_count": body.get("brick_count"),
                        "stable": body.get("stable"),
                        "final_stable": metadata.get("final_stable", body.get("stable")),
                        "termination_reason": metadata.get("termination_reason"),
                        "rejections": metadata.get("rejections"),
                        "rollbacks": metadata.get("rollbacks"),
                        "metadata": metadata,
                        "bricks": body.get("bricks", ""),
                        **checks,
                    })
                else:
                    row["detail"] = body.get("detail") if isinstance(body, dict) else None
            except Exception as exc:
                row.update({
                    "wall_time_ms": _elapsed_ms(start),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
            rows.append(row)
    finally:
        client.close()
    return rows


def run_export_benchmark(core_rows: list[dict[str, Any]], *, run_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    successful_core = [row for row in core_rows if row.get("success")]
    for row in successful_core:
        export_row: dict[str, Any] = {
            "mode": "export",
            "run_mode": run_mode,
            "prompt_id": row.get("prompt_id"),
            "prompt": row.get("prompt"),
            "core_brick_count": row.get("brick_count"),
            "success": False,
        }
        try:
            bricks = parse_brick_sequence(row.get("bricks", ""))
            export_info = inspect_export(bricks, title=f"export-{row.get('prompt_id')}")
            export_row.update({
                "success": bool(export_info["export_success"]),
                "parse_valid": True,
                **export_info,
            })
        except Exception as exc:
            export_row.update({
                "parse_valid": False,
                "export_success": False,
                "export_error": repr(exc),
                "ldraw_header_present": False,
                "ldraw_part_line_count": 0,
                "ldraw_part_line_count_matches_bricks": False,
                "ldraw_text": "",
            })
        rows.append(export_row)
    return rows


def summarize_core(rows: list[dict[str, Any]], *, run_mode: str, limits: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    limits = limits or _benchmark_limits(
        quick_smoke=False,
        max_bricks_per_sample=None,
        sample_timeout_s=None,
        stability_check_interval=None,
    )
    successes = [row for row in rows if row.get("success")]
    return [{
        "run_mode": run_mode,
        **_limit_row_fields(limits),
        "prompt_count": len(rows),
        "success_count": len(successes),
        "success_rate": _rate(row.get("success") for row in rows),
        "stable_rate": _rate(row.get("stable") for row in successes),
        "final_stable_rate": _rate(row.get("final_stable") for row in successes),
        "parse_valid_rate": _rate(row.get("parse_valid") for row in successes),
        "collision_free_rate": _rate(row.get("collision_free") for row in successes),
        "recomputed_stable_rate": _rate(row.get("recomputed_stable") for row in successes),
        "export_success_rate": _rate(row.get("export_success") for row in successes),
        "avg_brick_count": _safe_mean(row.get("brick_count") for row in successes),
        "avg_wall_time_ms": _safe_mean(row.get("wall_time_ms") for row in successes),
        "p50_wall_time_ms": _safe_p50(row.get("wall_time_ms") for row in successes),
        "avg_generation_time_ms": _safe_mean(row.get("generation_time_ms") for row in successes),
        "avg_rejections": _safe_mean(row.get("rejections") for row in successes),
        "avg_rollbacks": _safe_mean(row.get("rollbacks") for row in successes),
        "hit_max_bricks_rate": _rate(row.get("hit_max_bricks") for row in successes),
        "hit_max_seconds_rate": _rate(row.get("hit_max_seconds") for row in successes),
    }]


def summarize_bon(rows: list[dict[str, Any]], *, ns: list[int], limits: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    limits = limits or _benchmark_limits(
        quick_smoke=False,
        max_bricks_per_sample=None,
        sample_timeout_s=None,
        stability_check_interval=None,
    )
    summary: list[dict[str, Any]] = []
    for n in ns:
        subset = [row for row in rows if row.get("n") == n]
        successes = [row for row in subset if row.get("success")]
        summary.append({
            "n": n,
            **_limit_row_fields(limits),
            "prompt_count": len(subset),
            "success_count": len(successes),
            "success_rate": _rate(row.get("success") for row in subset),
            "stable_rate": _safe_mean(row.get("stable_rate") for row in successes),
            "picked_stable_rate": _rate(row.get("stable") for row in successes),
            "final_stable_rate": _rate(row.get("final_stable") for row in successes),
            "recomputed_stable_rate": _rate(row.get("recomputed_stable") for row in successes),
            "export_success_rate": _rate(row.get("export_success") for row in successes),
            "avg_brick_count": _safe_mean(row.get("brick_count") for row in successes),
            "avg_wall_time_ms": _safe_mean(row.get("wall_time_ms") for row in successes),
            "p50_wall_time_ms": _safe_p50(row.get("wall_time_ms") for row in successes),
            "avg_generation_time_ms": _safe_mean(row.get("generation_time_ms") for row in successes),
            "hit_max_bricks_rate": _rate(row.get("hit_max_bricks") for row in successes),
            "hit_max_seconds_rate": _rate(row.get("hit_max_seconds") for row in successes),
        })
    return summary


def summarize_stable_only(rows: list[dict[str, Any]], *, run_mode: str) -> list[dict[str, Any]]:
    successes = [row for row in rows if row.get("success")]
    return [{
        "run_mode": run_mode,
        "prompt_count": len(rows),
        "http_200_count": sum(1 for row in rows if row.get("status_code") == 200),
        "http_422_count": sum(1 for row in rows if row.get("status_code") == 422),
        "http_504_count": sum(1 for row in rows if row.get("status_code") == 504),
        "other_status_count": sum(1 for row in rows if row.get("status_code") not in (200, 422, 504)),
        "accepted_rate": _rate(row.get("status_code") == 200 for row in rows),
        "final_stable_rate_among_accepted": _rate(row.get("final_stable") for row in successes),
        "parse_valid_rate_among_accepted": _rate(row.get("parse_valid") for row in successes),
        "export_success_rate_among_accepted": _rate(row.get("export_success") for row in successes),
        "avg_wall_time_ms": _safe_mean(row.get("wall_time_ms") for row in rows),
    }]


def summarize_export(rows: list[dict[str, Any]], *, run_mode: str) -> list[dict[str, Any]]:
    return [{
        "run_mode": run_mode,
        "export_attempt_count": len(rows),
        "export_success_count": sum(1 for row in rows if row.get("export_success")),
        "export_success_rate": _rate(row.get("export_success") for row in rows),
        "header_presence_rate": _rate(row.get("ldraw_header_present") for row in rows),
        "part_line_count_match_rate": _rate(row.get("ldraw_part_line_count_matches_bricks") for row in rows),
        "avg_export_time_ms": _safe_mean(row.get("export_time_ms") for row in rows),
    }]


def _write_empty_outputs(
    run_dir: Path,
    modes: tuple[str, ...],
    *,
    run_mode: str,
    ns: list[int],
    limits: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    summaries: dict[str, list[dict[str, Any]]] = {}
    if "core" in modes:
        _write_jsonl(run_dir / "core_raw.jsonl", [])
        summaries["core"] = summarize_core([], run_mode=run_mode, limits=limits)
        _write_csv(run_dir / "core_summary.csv", summaries["core"], list(summaries["core"][0].keys()))
        _write_summary_md(run_dir / "core_summary.md", "Core Benchmark Summary", "No core rows were run.", summaries["core"], list(summaries["core"][0].keys()))
    if "bon" in modes:
        _write_jsonl(run_dir / "bon_raw.jsonl", [])
        summaries["bon"] = summarize_bon([], ns=ns, limits=limits)
        _write_csv(run_dir / "bon_summary.csv", summaries["bon"], list(summaries["bon"][0].keys()) if summaries["bon"] else ["n"])
        _write_summary_md(run_dir / "bon_summary.md", "Best-of-N Benchmark Summary", "No Best-of-N rows were run.", summaries["bon"], list(summaries["bon"][0].keys()) if summaries["bon"] else ["n"])
    if "stable-only" in modes:
        _write_jsonl(run_dir / "stable_only_raw.jsonl", [])
        summaries["stable-only"] = summarize_stable_only([], run_mode=run_mode)
        _write_csv(run_dir / "stable_only_summary.csv", summaries["stable-only"], list(summaries["stable-only"][0].keys()))
        _write_summary_md(run_dir / "stable_only_summary.md", "Stable-Only Route Benchmark Summary", "No stable-only rows were run.", summaries["stable-only"], list(summaries["stable-only"][0].keys()))
    if "export" in modes:
        _write_jsonl(run_dir / "export_raw.jsonl", [])
        summaries["export"] = summarize_export([], run_mode=run_mode)
        _write_csv(run_dir / "export_summary.csv", summaries["export"], list(summaries["export"][0].keys()))
        _write_summary_md(run_dir / "export_summary.md", "LDraw Export Benchmark Summary", "No export rows were run.", summaries["export"], list(summaries["export"][0].keys()))
    return summaries


def write_report(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    prompt_count: int,
    modes: tuple[str, ...],
    summaries: dict[str, list[dict[str, Any]]],
    limits: dict[str, Any],
    skip_reason: str | None = None,
    plot_paths: list[Path] | None = None,
) -> None:
    run_mode = metadata["environment"]["run_mode"]
    comparison = metadata.get("comparison", {}) or {}
    pipeline_meta = metadata.get("pipeline", {}) or {}
    lines = [
        "# LEGOGen Benchmark Report",
        "",
        f"- Run mode: `{run_mode}`",
        f"- LEGOGEN_DEV: `{metadata['environment']['LEGOGEN_DEV_raw']}`",
        f"- Prompts: `{prompt_count}`",
        f"- Modes: `{', '.join(modes)}`",
        f"- Git commit: `{metadata['git']['commit'] or 'unknown'}`",
        f"- Comparison label: `{comparison.get('label') or 'unknown'}`",
        f"- Brick model ID: `{comparison.get('model_id') or metadata['models']['brick_model_id']}`",
        f"- Brick checkpoint dir: `{comparison.get('checkpoint_dir') or metadata['models']['brick_checkpoint_dir']}`",
        f"- Visible CUDA devices: `{metadata['torch'].get('device_count', 0)}`",
    ]
    model_devices = pipeline_meta.get("model_devices") or []
    if model_devices:
        lines.append(f"- Model devices: `{', '.join(model_devices)}`")
        lines.append(f"- Model parallel active: `{pipeline_meta.get('model_parallel_active', False)}`")
    if run_mode == "dev-mock":
        lines.append("- Note: this is a dev/mock smoke result, not a real performance measurement.")
    if limits["quick_smoke"]:
        lines.append("- Note: this is a capped smoke run, not a full performance measurement.")
    elif limits["capped_generation"]:
        lines.append("- Note: generation caps are enabled; this is not a full uncapped performance measurement.")
    if pipeline_meta.get("model_parallel_active"):
        lines.append("- Note: multi-GPU model sharding is active for this run.")
    elif metadata["torch"].get("device_count", 0) > 1:
        lines.append("- Note: multiple CUDA devices were visible, but the loaded model did not shard across them.")
    if limits["capped_generation"]:
        lines.extend([
            f"- Max bricks per sample: `{limits['max_bricks_per_sample']}`",
            f"- Sample timeout seconds: `{limits['sample_timeout_s']}`",
            f"- Stability check interval: `{limits['stability_check_interval']}`",
        ])
    if skip_reason:
        lines.extend(["", f"Benchmark generation was skipped: {skip_reason}"])
    for name in ("core", "bon", "stable-only", "export"):
        if name in summaries and summaries[name]:
            lines.extend(["", f"## {name}", ""])
            first = summaries[name][0]
            for key, value in first.items():
                lines.append(f"- {key}: `{value}`")
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for path in plot_paths:
            lines.append(f"- `{path.relative_to(run_dir)}`")
    lines.append("")
    (run_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    *,
    prompts_path: Path = DEFAULT_PROMPTS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    timestamp: str | None = None,
    limit_prompts: int | None = None,
    modes: tuple[str, ...] = ALL_MODES,
    bon_ns: list[int] | None = None,
    bon_strategy: str = "cluster",
    comparison_label: str | None = None,
    allow_base_model: bool = False,
    quick_smoke: bool = False,
    max_bricks_per_sample: int | None = None,
    sample_timeout_s: float | None = None,
    stability_check_interval: int | None = None,
) -> Path:
    bon_ns = bon_ns or DEFAULT_NS
    if quick_smoke:
        if modes == ALL_MODES:
            modes = QUICK_SMOKE_MODES
        if limit_prompts is None:
            limit_prompts = QUICK_SMOKE_LIMIT_PROMPTS
        if max_bricks_per_sample is None:
            max_bricks_per_sample = QUICK_SMOKE_MAX_BRICKS
        if sample_timeout_s is None:
            sample_timeout_s = QUICK_SMOKE_SAMPLE_TIMEOUT_S
        if stability_check_interval is None:
            stability_check_interval = QUICK_SMOKE_STABILITY_CHECK_INTERVAL
    limits = _benchmark_limits(
        quick_smoke=quick_smoke,
        max_bricks_per_sample=max_bricks_per_sample,
        sample_timeout_s=sample_timeout_s,
        stability_check_interval=stability_check_interval,
    )
    timestamp = timestamp or _now_timestamp()
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(prompts_path, limit=limit_prompts)
    (run_dir / "prompts_used.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    config = {
        "prompts_path": str(prompts_path),
        "limit_prompts": limit_prompts,
        "modes": list(modes),
        "bon_ns": bon_ns,
        "bon_strategy": bon_strategy,
        "comparison_label": comparison_label or default_comparison_label(),
        "allow_base_model": allow_base_model,
        **limits,
    }
    _write_json(run_dir / "benchmark_config.json", config)

    metadata = collect_environment_metadata(limits)
    metadata["comparison"] = {
        "label": comparison_label or default_comparison_label(),
        "model_id": BRICK_MODEL_NAME,
        "checkpoint_dir": str(BRICK_CHECKPOINT_DIR),
        "allow_base_model": allow_base_model,
    }
    _write_json(run_dir / "environment_metadata.json", metadata)
    run_mode = metadata["environment"]["run_mode"]

    skip_reason = real_skip_reason(metadata, allow_base_model=allow_base_model)
    if skip_reason:
        summaries = _write_empty_outputs(run_dir, modes, run_mode=run_mode, ns=bon_ns, limits=limits)
        write_report(
            run_dir,
            metadata=metadata,
            prompt_count=len(prompts),
            modes=modes,
            summaries=summaries,
            limits=limits,
            skip_reason=skip_reason,
        )
        return run_dir

    try:
        from backend.inference.brick_pipeline import get_brick_pipeline

        pipeline = get_brick_pipeline()
        metadata["pipeline"] = collect_pipeline_metadata(
            pipeline,
            comparison_label=metadata["comparison"]["label"],
        )
        _write_json(run_dir / "environment_metadata.json", metadata)
    except Exception as exc:
        skip_reason = f"Could not initialize brick pipeline: {exc!r}"
        summaries = _write_empty_outputs(run_dir, modes, run_mode=run_mode, ns=bon_ns, limits=limits)
        write_report(
            run_dir,
            metadata=metadata,
            prompt_count=len(prompts),
            modes=modes,
            summaries=summaries,
            limits=limits,
            skip_reason=skip_reason,
        )
        return run_dir

    summaries: dict[str, list[dict[str, Any]]] = {}
    core_rows: list[dict[str, Any]] = []

    if "core" in modes or "export" in modes:
        core_rows = run_core_benchmark(pipeline, prompts, run_mode=run_mode, limits=limits)
        if "core" in modes:
            _write_jsonl(run_dir / "core_raw.jsonl", core_rows)
            summaries["core"] = summarize_core(core_rows, run_mode=run_mode, limits=limits)
            _write_csv(run_dir / "core_summary.csv", summaries["core"], list(summaries["core"][0].keys()))
            _write_summary_md(
                run_dir / "core_summary.md",
                "Core Benchmark Summary",
                "n=1 text generation through the current BrickPipeline.generate path. Capped runs are smoke tests, not full performance results.",
                summaries["core"],
                list(summaries["core"][0].keys()),
            )

    if "bon" in modes:
        bon_rows = run_bon_benchmark(
            pipeline,
            prompts,
            ns=bon_ns,
            strategy=bon_strategy,
            run_mode=run_mode,
            limits=limits,
        )
        _write_jsonl(run_dir / "bon_raw.jsonl", bon_rows)
        summaries["bon"] = summarize_bon(bon_rows, ns=bon_ns, limits=limits)
        _write_csv(run_dir / "bon_summary.csv", summaries["bon"], list(summaries["bon"][0].keys()))
        _write_summary_md(
            run_dir / "bon_summary.md",
            "Best-of-N Benchmark Summary",
            "Calls BrickPipeline.generate_best_of_n for each prompt and n. Capped runs are smoke tests, not full performance results.",
            summaries["bon"],
            list(summaries["bon"][0].keys()),
        )

    if "stable-only" in modes:
        stable_rows = run_stable_only_benchmark(prompts, run_mode=run_mode)
        _write_jsonl(run_dir / "stable_only_raw.jsonl", stable_rows)
        summaries["stable-only"] = summarize_stable_only(stable_rows, run_mode=run_mode)
        _write_csv(run_dir / "stable_only_summary.csv", summaries["stable-only"], list(summaries["stable-only"][0].keys()))
        _write_summary_md(
            run_dir / "stable_only_summary.md",
            "Stable-Only Route Benchmark Summary",
            "Route-level require_stable=True results. This measures HTTP 200/422/504 behavior; it is not a retry-until-stable benchmark.",
            summaries["stable-only"],
            list(summaries["stable-only"][0].keys()),
        )

    if "export" in modes:
        export_rows = run_export_benchmark(core_rows, run_mode=run_mode)
        _write_jsonl(run_dir / "export_raw.jsonl", export_rows)
        summaries["export"] = summarize_export(export_rows, run_mode=run_mode)
        _write_csv(run_dir / "export_summary.csv", summaries["export"], list(summaries["export"][0].keys()))
        _write_summary_md(
            run_dir / "export_summary.md",
            "LDraw Export Benchmark Summary",
            "LDraw export for every successful core output.",
            summaries["export"],
            list(summaries["export"][0].keys()),
        )

    plot_paths: list[Path] = []
    if "bon" in modes:
        try:
            from scripts.plot_benchmark_results import generate_plots

            plot_paths = generate_plots(run_dir)
        except Exception as exc:
            (run_dir / "plot_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")

    write_report(
        run_dir,
        metadata=metadata,
        prompt_count=len(prompts),
        modes=modes,
        summaries=summaries,
        limits=limits,
        plot_paths=plot_paths,
    )
    return run_dir


def _parse_modes(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return ALL_MODES
    invalid = sorted(set(values) - set(ALL_MODES))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown benchmark mode(s): {', '.join(invalid)}")
    modes = tuple(dict.fromkeys(values))
    if "export" in modes and "core" not in modes:
        modes = tuple([*modes, "core"])
    return modes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LEGOGen brick-coordinate benchmarks.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="Prompt file; comments and blank lines are ignored.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory that will contain <timestamp> run folders.")
    parser.add_argument("--timestamp", type=str, default=None, help="Run directory name. Defaults to local YYYYmmdd_HHMMSS.")
    parser.add_argument("--limit-prompts", type=int, default=None, help="Use the first N prompts only, useful for smoke runs.")
    parser.add_argument("--modes", nargs="+", default=None, help="Modes to run: all, core, bon, stable-only, export.")
    parser.add_argument("--bon-ns", type=int, nargs="+", default=DEFAULT_NS, help="Best-of-N values to sweep.")
    parser.add_argument("--bon-strategy", choices=["rank", "cluster"], default="cluster", help="Selection strategy passed to generate_best_of_n.")
    parser.add_argument("--comparison-label", type=str, default=None, help="Optional label for the benchmark target, useful when comparing checkpoints across runs.")
    parser.add_argument("--allow-base-model", action="store_true", help="Allow LEGOGEN_DEV=0 runs without the Stage 2 LoRA adapter present.")
    parser.add_argument("--quick-smoke", action="store_true", help="Run a capped core+export smoke benchmark, not a full performance run.")
    parser.add_argument("--max-bricks-per-sample", type=_positive_int, default=None, help="Cap each generated sample to this many bricks.")
    parser.add_argument("--sample-timeout-s", type=_positive_float, default=None, help="Cooperative timeout for each generated sample.")
    parser.add_argument("--stability-check-interval", type=_positive_int, default=None, help="Check stability and rollback every N accepted bricks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    mode_values = args.modes or (list(QUICK_SMOKE_MODES) if args.quick_smoke else ["all"])
    modes = _parse_modes(mode_values)
    run_dir = run_benchmark(
        prompts_path=args.prompts,
        output_root=args.output_root,
        timestamp=args.timestamp,
        limit_prompts=args.limit_prompts,
        modes=modes,
        bon_ns=args.bon_ns,
        bon_strategy=args.bon_strategy,
        comparison_label=args.comparison_label,
        allow_base_model=args.allow_base_model,
        quick_smoke=args.quick_smoke,
        max_bricks_per_sample=args.max_bricks_per_sample,
        sample_timeout_s=args.sample_timeout_s,
        stability_check_interval=args.stability_check_interval,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
