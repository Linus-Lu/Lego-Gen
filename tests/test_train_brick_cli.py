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
        "--learning-rate",
        "0.0007",
        "--lr-scheduler-type",
        "linear",
        "--weight-decay",
        "0.01",
        "--max-grad-norm",
        "0.75",
        "--optim",
        "adamw_torch_fused",
        "--max-seq-length",
        "2048",
        "--loss-chunk-tokens",
        "512",
        "--batch-size",
        "3",
        "--eval-batch-size",
        "2",
        "--gradient-accumulation-steps",
        "2",
        "--tokenized-cache-dir",
        str(tmp_path / "token-cache"),
        "--rebuild-tokenized-cache",
        "--no-gradient-checkpointing",
        "--torch-dtype",
        "bfloat16",
        "--attn-implementation",
        "sdpa",
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.1",
        "--lora-target-modules",
        "q_proj,k_proj,v_proj",
        "--no-dora",
        "--no-rslora",
        "--seed",
        "123",
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
    assert args.learning_rate == 0.0007
    assert args.lr_scheduler_type == "linear"
    assert args.weight_decay == 0.01
    assert args.max_grad_norm == 0.75
    assert args.optim == "adamw_torch_fused"
    assert args.max_seq_length == 2048
    assert args.loss_chunk_tokens == 512
    assert args.batch_size == 3
    assert args.eval_batch_size == 2
    assert args.gradient_accumulation_steps == 2
    assert args.tokenized_cache_dir == tmp_path / "token-cache"
    assert args.rebuild_tokenized_cache is True
    assert args.no_gradient_checkpointing is True
    assert args.torch_dtype == "bfloat16"
    assert args.attn_implementation == "sdpa"
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.1
    assert args.lora_target_modules == "q_proj,k_proj,v_proj"
    assert args.no_dora is True
    assert args.no_rslora is True
    assert args.seed == 123
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


def test_parse_lora_targets_and_torch_dtype():
    assert train_brick._parse_lora_target_modules("q_proj, v_proj,,") == ["q_proj", "v_proj"]
    assert train_brick._resolve_torch_dtype("auto") == "auto"

    with pytest.raises(SystemExit):
        train_brick._parse_lora_target_modules(" , ")


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


class _TinyTokenizer:
    name_or_path = "tiny-tokenizer"
    chat_template = "tiny-template"
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 128

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        return_dict,
        truncation=False,
        max_length=None,
    ):
        assert tokenize is True
        assert return_dict is True
        text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        ids = [(ord(char) % 97) + 3 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_tokenized_cache_metadata_changes_with_data(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text('{"messages": []}\n', encoding="utf-8")
    test_path.write_text('{"messages": []}\n', encoding="utf-8")

    metadata_a = train_brick._tokenized_cache_metadata(
        train_path=train_path,
        test_path=test_path,
        tokenizer=_TinyTokenizer(),
        max_seq_length=2048,
        train_samples=10,
        eval_samples=5,
        train_include_tail=2,
        seed=42,
    )
    train_path.write_text('{"messages": []}\n{"messages": []}\n', encoding="utf-8")
    metadata_b = train_brick._tokenized_cache_metadata(
        train_path=train_path,
        test_path=test_path,
        tokenizer=_TinyTokenizer(),
        max_seq_length=2048,
        train_samples=10,
        eval_samples=5,
        train_include_tail=2,
        seed=42,
    )

    assert train_brick._tokenized_cache_key(metadata_a) != train_brick._tokenized_cache_key(metadata_b)


def test_tokenized_cache_metadata_changes_with_max_seq_length(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_path.write_text('{"messages": []}\n', encoding="utf-8")
    test_path.write_text('{"messages": []}\n', encoding="utf-8")

    common = dict(
        train_path=train_path,
        test_path=test_path,
        tokenizer=_TinyTokenizer(),
        train_samples=10,
        eval_samples=5,
        train_include_tail=2,
        seed=42,
    )
    metadata_a = train_brick._tokenized_cache_metadata(max_seq_length=2048, **common)
    metadata_b = train_brick._tokenized_cache_metadata(max_seq_length=4096, **common)

    assert train_brick._tokenized_cache_key(metadata_a) != train_brick._tokenized_cache_key(metadata_b)


def test_tokenized_cache_loads_without_retokenizing(tmp_path):
    train_ds = datasets.Dataset.from_dict({
        "messages": [[
            {"role": "user", "content": "make a red house"},
            {"role": "assistant", "content": "2x4 (0,0,0) #C91A09\nDONE"},
        ]],
    })
    eval_ds = datasets.Dataset.from_dict({
        "messages": [[
            {"role": "user", "content": "make a blue car"},
            {"role": "assistant", "content": "2x4 (0,0,0) #0055BF\nDONE"},
        ]],
    })
    metadata = {"version": 1, "case": "unit"}
    tokenizer = _TinyTokenizer()

    train_a, eval_a = train_brick._load_or_create_tokenized_datasets(
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        max_seq_length=16,
        cache_root=tmp_path / "cache",
        metadata=metadata,
        rebuild=False,
    )
    tokenizer.apply_chat_template = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("re-tokenized"))
    train_b, eval_b = train_brick._load_or_create_tokenized_datasets(
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        max_seq_length=16,
        cache_root=tmp_path / "cache",
        metadata=metadata,
        rebuild=False,
    )

    assert train_a["input_ids"] == train_b["input_ids"]
    assert eval_a["input_ids"] == eval_b["input_ids"]
    assert train_b.column_names == ["input_ids"]
    assert max(len(ids) for ids in train_b["input_ids"]) <= 16
    assert max(len(ids) for ids in eval_b["input_ids"]) <= 16
