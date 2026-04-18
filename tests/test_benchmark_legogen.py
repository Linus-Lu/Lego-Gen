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
