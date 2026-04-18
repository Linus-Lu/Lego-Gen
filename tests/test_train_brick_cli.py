from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("trl")

from backend.training import train_brick


def test_train_brick_parser_exposes_v2_controls(tmp_path):
    args = train_brick.build_arg_parser().parse_args([
        "--data-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "out"),
        "--max-steps",
        "80",
        "--resume",
        "none",
        "--save-total-limit",
        "5",
        "--no-wandb",
    ])

    assert args.data_dir == tmp_path / "data"
    assert args.output_dir == str(tmp_path / "out")
    assert args.max_steps == 80
    assert args.resume == "none"
    assert args.save_total_limit == 5
    assert args.no_wandb is True


def test_resolve_resume_checkpoint_none_and_auto(tmp_path):
    assert train_brick._resolve_resume_checkpoint("none", str(tmp_path)) is None

    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-2").mkdir()

    assert train_brick._resolve_resume_checkpoint("auto", str(tmp_path)) == str(tmp_path / "checkpoint-10")
    assert train_brick._resolve_resume_checkpoint(str(tmp_path / "custom"), str(tmp_path)) == str(tmp_path / "custom")
