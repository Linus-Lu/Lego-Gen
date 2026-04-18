from pathlib import Path

import pytest

pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
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
        "--train-samples",
        "4096",
        "--eval-samples",
        "256",
        "--eval-steps",
        "100",
        "--save-steps",
        "200",
        "--warmup-steps",
        "20",
        "--gradient-accumulation-steps",
        "2",
        "--no-wandb",
    ])

    assert args.data_dir == tmp_path / "data"
    assert args.output_dir == str(tmp_path / "out")
    assert args.max_steps == 80
    assert args.resume == "none"
    assert args.save_total_limit == 5
    assert args.train_samples == 4096
    assert args.eval_samples == 256
    assert args.eval_steps == 100
    assert args.save_steps == 200
    assert args.warmup_steps == 20
    assert args.gradient_accumulation_steps == 2
    assert args.no_wandb is True


def test_resolve_resume_checkpoint_none_and_auto(tmp_path):
    assert train_brick._resolve_resume_checkpoint("none", str(tmp_path)) is None

    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-2").mkdir()

    assert train_brick._resolve_resume_checkpoint("auto", str(tmp_path)) == str(tmp_path / "checkpoint-10")
    assert train_brick._resolve_resume_checkpoint(str(tmp_path / "custom"), str(tmp_path)) == str(tmp_path / "custom")


def test_effective_warmup_steps_caps_short_gates():
    assert train_brick._effective_warmup_steps(None, 100) == 100
    assert train_brick._effective_warmup_steps(80, 100) == 8
    assert train_brick._effective_warmup_steps(80, 4) == 4

    with pytest.raises(SystemExit):
        train_brick._effective_warmup_steps(0, 100)


def test_select_dataset_subset_is_deterministic_and_bounded():
    ds = datasets.Dataset.from_dict({"value": list(range(20))})

    subset_a = train_brick._select_dataset_subset(ds, 5, seed=123, name="eval")
    subset_b = train_brick._select_dataset_subset(ds, 5, seed=123, name="eval")
    with_tail = train_brick._select_dataset_subset(
        ds,
        5,
        seed=123,
        name="train",
        include_tail=2,
    )
    full = train_brick._select_dataset_subset(ds, 25, seed=123, name="eval")

    assert len(subset_a) == 5
    assert subset_a["value"] == subset_b["value"]
    assert len(set(subset_a["value"])) == 5
    assert len(with_tail) == 5
    assert with_tail["value"][-2:] == [18, 19]
    assert len(full) == len(ds)

    with pytest.raises(SystemExit):
        train_brick._select_dataset_subset(ds, 0, seed=123, name="eval")
