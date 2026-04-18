import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import benchmark_legogen as bench


def test_validate_bricks_text_recomputes_current_guardrails():
    bricks = "2x4 (0,0,0) #C91A09\n2x4 (0,0,1) #FFFFFF"

    checks = bench.validate_bricks_text(bricks, title="Unit Test")

    assert checks["parse_valid"] is True
    assert checks["recomputed_brick_count"] == 2
    assert checks["collision_free"] is True
    assert checks["recomputed_stable"] is True
    assert checks["export_success"] is True
    assert checks["ldraw_header_present"] is True
    assert checks["ldraw_part_line_count"] == 2
    assert checks["ldraw_part_line_count_matches_bricks"] is True


def test_run_benchmark_writes_expected_dev_outputs(tmp_path, monkeypatch, reset_pipeline_singletons):
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a red house\n\na blue car\n", encoding="utf-8")

    run_dir = bench.run_benchmark(
        prompts_path=prompts,
        output_root=tmp_path / "runs",
        timestamp="dev-smoke",
        modes=("core", "bon", "stable-only", "export"),
        bon_ns=[1, 2],
        bon_strategy="rank",
    )

    expected = [
        "environment_metadata.json",
        "benchmark_config.json",
        "prompts_used.txt",
        "core_raw.jsonl",
        "core_summary.csv",
        "core_summary.md",
        "bon_raw.jsonl",
        "bon_summary.csv",
        "bon_summary.md",
        "stable_only_raw.jsonl",
        "stable_only_summary.csv",
        "stable_only_summary.md",
        "export_raw.jsonl",
        "export_summary.csv",
        "export_summary.md",
        "benchmark_report.md",
    ]
    for name in expected:
        assert (run_dir / name).exists(), name

    assert (run_dir / "plots" / "stable_rate_vs_n.svg").exists()
    assert (run_dir / "plots" / "brick_count_vs_n.svg").exists()
    assert (run_dir / "plots" / "latency_vs_n.svg").exists()

    core_rows = [
        json.loads(line)
        for line in (run_dir / "core_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(core_rows) == 2
    assert all(row["success"] for row in core_rows)
    assert all(row["parse_valid"] for row in core_rows)

    report = (run_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "dev/mock smoke result" in report


def test_stable_only_benchmark_records_route_422(monkeypatch, reset_pipeline_singletons):
    monkeypatch.setenv("LEGOGEN_DEV", "1")
    from backend.inference import brick_pipeline as bp

    def unstable_generate(self, caption, on_progress=None):
        return {
            "bricks": "2x4 (0,0,0) #C91A09",
            "brick_count": 1,
            "stable": False,
            "metadata": {
                "model_version": "mock",
                "generation_time_ms": 1,
                "rejections": 0,
                "rollbacks": 0,
                "final_stable": False,
            },
        }

    monkeypatch.setattr(bp.MockBrickPipeline, "generate", unstable_generate)

    rows = bench.run_stable_only_benchmark(["unstable prompt"], run_mode="dev-mock")

    assert len(rows) == 1
    assert rows[0]["status_code"] == 422
    assert rows[0]["success"] is False
    assert "stable" in rows[0]["detail"].lower()


def test_run_core_benchmark_passes_generation_caps():
    class StubPipeline:
        def __init__(self):
            self.calls = []

        def generate(self, caption, on_progress=None, *, max_bricks=None, max_seconds=None, stability_check_interval=None):
            self.calls.append({
                "caption": caption,
                "max_bricks": max_bricks,
                "max_seconds": max_seconds,
                "stability_check_interval": stability_check_interval,
            })
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {
                    "generation_time_ms": 1,
                    "termination_reason": "max_bricks",
                    "final_stable": True,
                    "rejections": 0,
                    "rollbacks": 0,
                    "outlines_enabled": True,
                    "palette_validation_enabled": True,
                    "requested_max_bricks": max_bricks,
                    "requested_max_seconds": max_seconds,
                    "requested_stability_check_interval": stability_check_interval,
                    "hit_max_bricks": True,
                    "hit_max_seconds": False,
                },
            }

    limits = bench._benchmark_limits(
        quick_smoke=True,
        max_bricks_per_sample=7,
        sample_timeout_s=1.5,
        stability_check_interval=3,
    )
    pipeline = StubPipeline()

    rows = bench.run_core_benchmark(
        pipeline,
        ["a red house"],
        run_mode="real",
        limits=limits,
    )

    assert pipeline.calls == [{
        "caption": "a red house",
        "max_bricks": 7,
        "max_seconds": 1.5,
        "stability_check_interval": 3,
    }]
    assert rows[0]["quick_smoke"] is True
    assert rows[0]["max_bricks_per_sample"] == 7
    assert rows[0]["sample_timeout_s"] == 1.5
    assert rows[0]["stability_check_interval"] == 3
    assert rows[0]["requested_max_bricks"] == 7
    assert rows[0]["hit_max_bricks"] is True


def test_quick_smoke_writes_capped_report_and_config(tmp_path, monkeypatch):
    from backend.inference import brick_pipeline as bp

    class StubPipeline:
        def generate(self, caption, on_progress=None, *, max_bricks=None, max_seconds=None, stability_check_interval=None):
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {
                    "generation_time_ms": 1,
                    "termination_reason": "max_bricks",
                    "final_stable": True,
                    "rejections": 0,
                    "rollbacks": 0,
                    "outlines_enabled": True,
                    "palette_validation_enabled": True,
                    "requested_max_bricks": max_bricks,
                    "requested_max_seconds": max_seconds,
                    "requested_stability_check_interval": stability_check_interval,
                    "hit_max_bricks": True,
                    "hit_max_seconds": False,
                },
            }

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a red house\n", encoding="utf-8")
    monkeypatch.setattr(bp, "get_brick_pipeline", lambda: StubPipeline())
    monkeypatch.setattr(
        bench,
        "collect_environment_metadata",
        lambda limits=None: {
            "environment": {
                "run_mode": "dev-mock",
                "LEGOGEN_DEV_raw": "1",
                "LEGOGEN_DEV_effective": True,
                "CUDA_VISIBLE_DEVICES": "0,1",
            },
            "git": {"commit": "test", "status_short": "", "dirty": False},
            "models": {
                "brick_model_id": "Qwen/Qwen3.5-4B",
                "brick_checkpoint_dir": "/tmp/checkpoint",
            },
            "torch": {
                "device_count": 2,
            },
            "benchmark_limits": limits,
        },
    )

    run_dir = bench.run_benchmark(
        prompts_path=prompts,
        output_root=tmp_path / "runs",
        timestamp="quick-smoke",
        quick_smoke=True,
    )

    config = json.loads((run_dir / "benchmark_config.json").read_text(encoding="utf-8"))
    assert config["quick_smoke"] is True
    assert config["modes"] == ["core", "export"]
    assert config["limit_prompts"] == 1
    assert config["max_bricks_per_sample"] == 24
    assert config["sample_timeout_s"] == 90.0
    assert config["stability_check_interval"] == 8

    core_rows = [
        json.loads(line)
        for line in (run_dir / "core_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert core_rows[0]["success"] is True
    assert core_rows[0]["quick_smoke"] is True
    assert core_rows[0]["max_bricks_per_sample"] == 24
    assert core_rows[0]["parse_valid"] is True
    assert core_rows[0]["collision_free"] is True
    assert core_rows[0]["export_success"] is True

    report = (run_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "capped smoke run" in report
    assert "not a full performance measurement" in report


def test_collect_pipeline_metadata_reports_model_parallel_and_caps_support():
    class StubModel:
        hf_device_map = {"layer0": "cuda:0", "layer1": "cuda:1", "lm_head": "cuda:1"}

    class StubPipeline:
        def __init__(self):
            self.model = StubModel()

        def generate(self, caption, on_progress=None, *, max_bricks=None, max_seconds=None, stability_check_interval=None):
            return {}

        def generate_best_of_n(self, caption, n=1, strategy="cluster", on_progress=None, *, max_bricks=None, max_seconds=None, stability_check_interval=None):
            return {}

    metadata = bench.collect_pipeline_metadata(StubPipeline(), comparison_label="checkpoint-19000")

    assert metadata["comparison_label"] == "checkpoint-19000"
    assert metadata["pipeline_class"] == "StubPipeline"
    assert metadata["model_class"] == "StubModel"
    assert metadata["supports_generate_best_of_n"] is True
    assert metadata["supports_generation_limits"] is True
    assert metadata["hf_device_map_devices"] == ["cuda:0", "cuda:1"]
    assert metadata["model_devices"] == ["cuda:0", "cuda:1"]
    assert metadata["model_parallel_active"] is True


def test_run_benchmark_persists_comparison_label_and_model_placement(tmp_path, monkeypatch):
    class StubModel:
        hf_device_map = {"layer0": "cuda:0", "layer1": "cuda:1"}

    class StubPipeline:
        def __init__(self):
            self.model = StubModel()

        def generate(self, caption, on_progress=None, *, max_bricks=None, max_seconds=None, stability_check_interval=None):
            return {
                "bricks": "2x4 (0,0,0) #C91A09",
                "brick_count": 1,
                "stable": True,
                "metadata": {
                    "generation_time_ms": 1,
                    "termination_reason": "done",
                    "final_stable": True,
                    "rejections": 0,
                    "rollbacks": 0,
                    "outlines_enabled": True,
                    "palette_validation_enabled": True,
                    "requested_max_bricks": max_bricks,
                    "requested_max_seconds": max_seconds,
                    "requested_stability_check_interval": stability_check_interval,
                    "hit_max_bricks": False,
                    "hit_max_seconds": False,
                },
            }

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a red house\n", encoding="utf-8")
    monkeypatch.setattr(bench, "collect_environment_metadata", lambda limits=None: {
        "environment": {
            "run_mode": "real",
            "LEGOGEN_DEV_raw": "0",
            "LEGOGEN_DEV_effective": False,
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
        "git": {"commit": "test", "status_short": "", "dirty": False},
        "models": {
            "brick_model_id": "Qwen/Qwen3.5-4B",
            "brick_checkpoint_dir": "/tmp/checkpoint-19000",
            "brick_adapter_config_present": True,
        },
        "torch": {
            "cuda_available": True,
            "device_count": 2,
            "gpu_names": ["GPU0", "GPU1"],
        },
        "benchmark_limits": limits,
    })
    monkeypatch.setattr(bench, "real_skip_reason", lambda metadata, allow_base_model=False: None)
    monkeypatch.setattr("backend.inference.brick_pipeline.get_brick_pipeline", lambda: StubPipeline())

    run_dir = bench.run_benchmark(
        prompts_path=prompts,
        output_root=tmp_path / "runs",
        timestamp="comparison-run",
        modes=("core",),
        comparison_label="checkpoint-19000",
    )

    config = json.loads((run_dir / "benchmark_config.json").read_text(encoding="utf-8"))
    assert config["comparison_label"] == "checkpoint-19000"

    metadata = json.loads((run_dir / "environment_metadata.json").read_text(encoding="utf-8"))
    assert metadata["comparison"]["label"] == "checkpoint-19000"
    assert metadata["pipeline"]["model_parallel_active"] is True
    assert metadata["pipeline"]["model_devices"] == ["cuda:0", "cuda:1"]

    report = (run_dir / "benchmark_report.md").read_text(encoding="utf-8")
    assert "Comparison label: `checkpoint-19000`" in report
    assert "multi-GPU model sharding is active" in report
