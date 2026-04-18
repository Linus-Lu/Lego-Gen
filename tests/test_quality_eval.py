import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import eval_generation_quality as quality


def test_run_quality_eval_writes_outputs_and_passes_caps(tmp_path, monkeypatch):
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
                "bricks": (
                    "2x4 (0,0,0) #C91A09\n"
                    "2x4 (0,0,1) #FFFFFF\n"
                    "2x4 (0,0,2) #F2CD37"
                ),
                "brick_count": 3,
                "stable": True,
                "metadata": {
                    "generation_time_ms": 1,
                    "termination_reason": "done",
                    "hit_done": True,
                    "hit_max_bricks": False,
                    "hit_max_seconds": False,
                    "rollbacks": 0,
                    "rejections": 0,
                },
            }

    prompts = tmp_path / "quality.jsonl"
    prompts.write_text(
        json.dumps({
            "id": "house",
            "prompt": "a red house with white walls and a yellow roof",
            "expected_colors": ["C91A09", "FFFFFF", "F2CD37"],
            "min_distinct_colors": 3,
            "min_bricks": 3,
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        quality,
        "collect_environment_metadata",
        lambda limits=None: {
            "environment": {
                "run_mode": "dev-mock",
                "LEGOGEN_DEV_raw": "1",
                "LEGOGEN_DEV_effective": True,
            },
            "git": {"commit": "test", "status_short": "", "dirty": False},
            "benchmark_limits": limits,
        },
    )
    pipeline = StubPipeline()

    run_dir = quality.run_quality_eval(
        prompts_path=prompts,
        output_root=tmp_path / "runs",
        timestamp="quality-smoke",
        max_bricks_per_sample=12,
        sample_timeout_s=3.0,
        stability_check_interval=4,
        pipeline=pipeline,
    )

    assert pipeline.calls == [{
        "caption": "a red house with white walls and a yellow roof",
        "max_bricks": 12,
        "max_seconds": 3.0,
        "stability_check_interval": 4,
    }]
    for name in [
        "quality_config.json",
        "environment_metadata.json",
        "quality_raw.jsonl",
        "quality_summary.csv",
        "quality_summary.md",
        "quality_report.md",
    ]:
        assert (run_dir / name).exists(), name

    row = json.loads((run_dir / "quality_raw.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["expected_colors_all_present"] is True
    assert row["expected_color_coverage"] == 1.0
    assert row["meets_min_distinct_colors"] is True
    assert row["hit_done"] is True

    report = (run_dir / "quality_report.md").read_text(encoding="utf-8")
    assert "dev/mock results are smoke checks only" in report


def test_read_quality_prompts_rejects_missing_prompt(tmp_path):
    prompts = tmp_path / "bad.jsonl"
    prompts.write_text('{"id":"bad"}\n', encoding="utf-8")

    try:
        quality.read_quality_prompts(prompts)
    except ValueError as exc:
        assert "missing required 'prompt'" in str(exc)
    else:
        raise AssertionError("expected ValueError")
