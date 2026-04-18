"""Fine-tune Qwen3.5-4B with LoRA for text → colored brick sequence generation.

Features ported from train_unified.py:
  - Structure-aware loss weighting (upweight coordinates/dimensions)
  - Curriculum ordering (untruncated samples first)
  - Chunked cross-entropy (avoids OOM on 248K vocab)
"""

import argparse
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    TrainerCallback,
)
from trl import SFTTrainer

# Check if SFTConfig exists (TRL >= 0.12)
try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - transformers normally depends on it
    FileLock = None

from backend.config import (
    BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, BRICK_LEARNING_RATE,
    BRICK_BATCH_SIZE, BRICK_GRADIENT_ACCUMULATION, BRICK_MAX_SEQ_LENGTH,
    BRICK_NUM_EPOCHS, BRICK_LORA_R, BRICK_LORA_ALPHA, BRICK_LORA_DROPOUT,
    BRICK_TRAINING_DATA, USE_BF16,
)

LOSS_CHUNK_TOKENS = 1024
DEFAULT_EVAL_SAMPLES = 512
DEFAULT_SEED = 42
TOKENIZED_CACHE_VERSION = 2
CACHE_METADATA_FILE = "legogen_tokenized_cache.json"


# ── Structure-aware loss for brick format ─────────────────────────────────

class BrickStructureWeights:
    """Per-token loss weights for brick coordinate format.

    Brick format: "2x4 (5,3,0) #C91A09"
      - Boilerplate: parentheses, commas, 'x', '#', newlines, spaces → weight 0.1
      - Coordinates/dimensions (the actual numbers) → weight 1.0 (default)
      - Structural tokens: dimension combos like "2x4", coordinate digits → weight 3.0
    """

    def __init__(self, tokenizer, boilerplate_weight=0.1, structure_weight=3.0):
        self.boilerplate_weight = boilerplate_weight
        self.structure_weight = structure_weight

        # Boilerplate: syntax characters that repeat in every brick line
        self.boilerplate_ids: set[int] = set()
        for char in ["(", ")", ",", "x", "#", "\n", " ", "(,", "),",
                     "  ", "    ", "\n\n"]:
            ids = tokenizer.encode(char, add_special_tokens=False)
            self.boilerplate_ids.update(ids)

        # Structure: brick dimensions and coordinate patterns
        self.structure_ids: set[int] = set()
        for word in ["2x4", "4x2", "2x6", "6x2", "1x2", "2x1",
                     "1x4", "4x1", "1x6", "6x1", "1x8", "8x1",
                     "1x1", "2x2"]:
            ids = tokenizer.encode(word, add_special_tokens=False)
            self.structure_ids.update(ids)

        # Remove overlap (structure wins)
        self.boilerplate_ids -= self.structure_ids
        self._id_cache = {}

        print(f"[BrickStructureWeights] boilerplate token IDs: {len(self.boilerplate_ids)}, "
              f"structure token IDs: {len(self.structure_ids)}")

    def _cached_ids(self, name: str, ids: set[int], device: torch.device):
        if not ids:
            return None
        key = (name, str(device))
        cached = self._id_cache.get(key)
        if cached is None:
            cached = torch.tensor(sorted(ids), device=device)
            self._id_cache[key] = cached
        return cached

    def get_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Return per-token weights. Shape matches labels. Vectorized."""
        weights = torch.ones_like(labels, dtype=torch.float32)

        bp_ids = self._cached_ids("boilerplate", self.boilerplate_ids, labels.device)
        if bp_ids is not None:
            bp_mask = (labels.unsqueeze(-1) == bp_ids).any(-1)
            weights[bp_mask] = self.boilerplate_weight

        st_ids = self._cached_ids("structure", self.structure_ids, labels.device)
        if st_ids is not None:
            st_mask = (labels.unsqueeze(-1) == st_ids).any(-1)
            weights[st_mask] = self.structure_weight

        # Masked tokens (-100) get weight 0
        weights[labels == -100] = 0.0
        return weights


# ── Curriculum dataset & sampler ──────────────────────────────────────────

def apply_curriculum_ordering(hf_dataset, max_seq_length: int):
    """Reorder HF dataset: untruncated samples first, then truncated.

    Returns the reordered dataset and the count of untruncated samples.
    """
    PROMPT_OVERHEAD = 80
    CHARS_PER_TOKEN = 3.0

    fits, truncated = [], []
    for i in range(len(hf_dataset)):
        messages = hf_dataset[i]["messages"]
        assistant_len = len(messages[-1]["content"]) if messages else 0
        est_tokens = int(assistant_len / CHARS_PER_TOKEN) + PROMPT_OVERHEAD
        if est_tokens <= max_seq_length:
            fits.append(i)
        else:
            truncated.append(i)

    # Reorder the HF dataset directly (preserves all HF Dataset methods)
    ordered_indices = fits + truncated
    reordered = hf_dataset.select(ordered_indices)

    print(f"  Curriculum: {len(fits)} untruncated, "
          f"{len(truncated)} truncated ({len(truncated)*100/max(1,len(ordered_indices)):.1f}%) "
          f"at max_seq_length={max_seq_length}")

    return reordered, len(fits)


class MemoryCleanupCallback(TrainerCallback):
    """Free GPU memory after evaluation."""

    def on_evaluate(self, args, state, control, **kwargs):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _inspect_params(cls):
    """Return set of parameter names accepted by cls.__init__."""
    try:
        return set(inspect.signature(cls.__init__).parameters.keys())
    except (ValueError, TypeError):
        return set()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help="Resume policy: 'none', 'auto', or an explicit checkpoint path.",
    )
    parser.add_argument("--output-dir", type=str, default=str(BRICK_CHECKPOINT_DIR),
                        help="Directory to save adapter checkpoints")
    parser.add_argument("--data-dir", type=Path, default=BRICK_TRAINING_DATA,
                        help="Directory containing train.jsonl and test.jsonl")
    parser.add_argument("--epochs", type=int, default=BRICK_NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many optimizer steps; useful for canary gates")
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Optional deterministic train subset size for fast gates")
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES,
                        help="Deterministic eval subset size; keeps step eval bounded")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Run bounded eval every N optimizer steps")
    parser.add_argument("--save-steps", type=int, default=200,
                        help="Save checkpoints every N optimizer steps")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log training metrics every N optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Requested LR warmup steps; capped to 10%% for max-step gates")
    parser.add_argument("--learning-rate", type=float, default=BRICK_LEARNING_RATE,
                        help="Optimizer learning rate")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        help="Transformers LR scheduler name")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Gradient clipping norm")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Transformers optimizer name")
    parser.add_argument("--max-seq-length", type=int, default=BRICK_MAX_SEQ_LENGTH,
                        help="Maximum token sequence length")
    parser.add_argument("--loss-chunk-tokens", type=int, default=LOSS_CHUNK_TOKENS,
                        help="Token chunk size for structure-weighted CE")
    parser.add_argument("--batch-size", type=int, default=BRICK_BATCH_SIZE,
                        help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        default=BRICK_GRADIENT_ACCUMULATION,
                        help="Gradient accumulation steps")
    parser.add_argument("--eval-accumulation-steps", type=int, default=1,
                        help="Move eval outputs to CPU this often when supported")
    parser.add_argument("--dataloader-num-workers", type=int, default=4,
                        help="Training dataloader worker processes")
    parser.add_argument("--group-by-length", action="store_true",
                        help="Bucket tokenized samples by length to reduce padding and DDP stragglers")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true",
                        help="Enable DDP unused-parameter detection; disabled by default for LoRA speed")
    parser.add_argument("--tokenized-cache-dir", type=Path, default=None,
                        help="Directory for reusable tokenized datasets; defaults to <data-dir>/.tokenized_cache")
    parser.add_argument("--rebuild-tokenized-cache", action="store_true",
                        help="Rebuild the matching tokenized dataset cache entry")
    parser.add_argument("--no-tokenized-cache", action="store_true",
                        help="Disable pre-tokenization cache and let SFTTrainer tokenize every run")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing; faster but uses more VRAM")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"],
                        default="auto", help="Model load dtype")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
        help="Transformers attention implementation",
    )
    parser.add_argument("--lora-r", type=int, default=BRICK_LORA_R,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=BRICK_LORA_ALPHA,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=BRICK_LORA_DROPOUT,
                        help="LoRA dropout")
    parser.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--no-dora", action="store_true",
                        help="Disable DoRA in the LoRA adapter")
    parser.add_argument("--no-rslora", action="store_true",
                        help="Disable rsLoRA in the LoRA adapter")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Training and deterministic subset seed")
    parser.add_argument("--save-total-limit", type=int, default=5,
                        help="How many recent checkpoints to retain")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    return parser


def _resolve_resume_checkpoint(resume: str | None, output_dir: str) -> str | None:
    policy = (resume or "none").strip()
    if policy.lower() in {"", "none", "false", "0"}:
        return None
    if policy.lower() != "auto":
        return policy

    ckpt_dir = Path(str(output_dir))
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        ckpt_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
        if p.name.split("-")[-1].isdigit() else 0,
    )
    return str(checkpoints[-1]) if checkpoints else None


def _validate_positive(name: str, value: int | None, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if value is None or value <= 0:
        raise SystemExit(f"--{name} must be positive")


def _effective_warmup_steps(max_steps: int | None, requested_warmup_steps: int) -> int:
    _validate_positive("warmup-steps", requested_warmup_steps)
    if max_steps is None:
        return requested_warmup_steps
    _validate_positive("max-steps", max_steps)
    # Canary gates should spend some steps at a useful learning rate instead of
    # ending entirely inside the default 100-step warmup.
    return min(requested_warmup_steps, max(1, max_steps // 10))


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise SystemExit(f"--{name} must be non-negative")


def _parse_lora_target_modules(value: str) -> list[str]:
    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise SystemExit("--lora-target-modules must include at least one module")
    return modules


def _resolve_torch_dtype(value: str):
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def _select_dataset_subset(
    dataset,
    limit: int | None,
    *,
    seed: int,
    name: str,
    include_tail: int = 0,
):
    _validate_positive(f"{name}-samples", limit, allow_none=True)
    if include_tail < 0:
        raise SystemExit("--include-tail must be non-negative")
    if limit is None or limit >= len(dataset):
        print(f"Using full {name} dataset: {len(dataset)} examples", flush=True)
        return dataset
    tail_count = min(include_tail, limit)
    head_count = limit - tail_count
    if tail_count:
        prefix_len = max(0, len(dataset) - tail_count)
        prefix = dataset.select(range(prefix_len))
        pieces = []
        if head_count:
            pieces.append(prefix.shuffle(seed=seed).select(range(head_count)))
        pieces.append(dataset.select(range(len(dataset) - tail_count, len(dataset))))
        subset = concatenate_datasets(pieces)
    else:
        subset = dataset.shuffle(seed=seed).select(range(limit))
    print(f"Using {name} subset: {len(subset)} / {len(dataset)} examples", flush=True)
    return subset


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenized_cache_root(data_dir: Path, explicit_cache_dir: Path | None) -> Path:
    return explicit_cache_dir if explicit_cache_dir is not None else data_dir / ".tokenized_cache"


def _tokenizer_fingerprint(tokenizer) -> dict:
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", ""),
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }


def _tokenized_cache_metadata(
    *,
    train_path: Path,
    test_path: Path,
    tokenizer,
    max_seq_length: int,
    train_samples: int | None,
    eval_samples: int | None,
    train_include_tail: int,
    seed: int,
) -> dict:
    return {
        "version": TOKENIZED_CACHE_VERSION,
        "model_name": BRICK_MODEL_NAME,
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "train_file": {
            "path": str(train_path.resolve()),
            "sha256": _file_sha256(train_path),
        },
        "test_file": {
            "path": str(test_path.resolve()),
            "sha256": _file_sha256(test_path),
        },
        "selection": {
            "seed": seed,
            "train_samples": train_samples,
            "eval_seed": seed + 1,
            "eval_samples": eval_samples,
            "train_include_tail": train_include_tail,
            "max_seq_length": max_seq_length,
        },
        "format": {
            "source_column": "messages",
            "tokenizer_call": "apply_chat_template(tokenize=True, return_dict=True, truncation=True, max_length=max_seq_length)",
        },
    }


def _tokenized_cache_key(metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _metadata_matches(cache_dir: Path, metadata: dict) -> bool:
    metadata_path = cache_dir / CACHE_METADATA_FILE
    if not metadata_path.exists():
        return False
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    except json.JSONDecodeError:
        return False


def _cache_lock(path: Path):
    if FileLock is None:
        return nullcontext()
    return FileLock(str(path))


def _unwrap_tokenizer_output(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        return value[0]
    return value


def _tokenize_chat_dataset(dataset, tokenizer, *, name: str, max_seq_length: int):
    def tokenize_example(example):
        processed = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = _unwrap_tokenizer_output(processed["input_ids"])
        output = {"input_ids": input_ids, "length": len(input_ids)}
        if "assistant_masks" in processed:
            output["assistant_masks"] = _unwrap_tokenizer_output(processed["assistant_masks"])
        return output

    return dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {name} dataset",
    )


def _load_or_create_tokenized_datasets(
    *,
    train_ds,
    eval_ds,
    tokenizer,
    max_seq_length: int,
    cache_root: Path,
    metadata: dict,
    rebuild: bool,
):
    cache_key = _tokenized_cache_key(metadata)
    cache_dir = cache_root / cache_key
    lock_path = cache_root / f"{cache_key}.lock"
    cache_root.mkdir(parents=True, exist_ok=True)

    with _cache_lock(lock_path):
        if cache_dir.exists() and not rebuild and _metadata_matches(cache_dir, metadata):
            print(f"Loading tokenized dataset cache: {cache_dir}", flush=True)
            tokenized = load_from_disk(str(cache_dir))
            return tokenized["train"], tokenized["eval"]

        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        print(f"Building tokenized dataset cache: {cache_dir}", flush=True)
        tokenized = DatasetDict({
            "train": _tokenize_chat_dataset(
                train_ds,
                tokenizer,
                name="train",
                max_seq_length=max_seq_length,
            ),
            "eval": _tokenize_chat_dataset(
                eval_ds,
                tokenizer,
                name="eval",
                max_seq_length=max_seq_length,
            ),
        })

        tmp_dir = cache_root / f".{cache_key}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tokenized.save_to_disk(str(tmp_dir))
        (tmp_dir / CACHE_METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_dir.replace(cache_dir)
        print(f"Saved tokenized dataset cache: {cache_dir}", flush=True)
        return tokenized["train"], tokenized["eval"]


def main() -> None:
    parser = build_arg_parser()
    # parse_args (not parse_known_args) so unknown flags surface as errors;
    # the silent-drop behaviour previously here hid typos in scripts.
    args = parser.parse_args()
    for flag_name in [
        "eval-steps",
        "save-steps",
        "logging-steps",
        "batch-size",
        "eval-batch-size",
        "gradient-accumulation-steps",
        "eval-accumulation-steps",
        "max-seq-length",
        "loss-chunk-tokens",
        "lora-r",
        "lora-alpha",
        "save-total-limit",
    ]:
        _validate_positive(flag_name, getattr(args, flag_name.replace("-", "_")))
    _validate_nonnegative("learning-rate", args.learning_rate)
    _validate_nonnegative("weight-decay", args.weight_decay)
    _validate_nonnegative("max-grad-norm", args.max_grad_norm)
    _validate_nonnegative("lora-dropout", args.lora_dropout)
    if args.lora_dropout > 1:
        raise SystemExit("--lora-dropout must be <= 1")
    if args.dataloader_num_workers < 0:
        raise SystemExit("--dataloader-num-workers must be non-negative")
    if args.save_steps % args.eval_steps != 0:
        raise SystemExit("--save-steps must be a multiple of --eval-steps")
    warmup_steps = _effective_warmup_steps(args.max_steps, args.warmup_steps)
    if warmup_steps != args.warmup_steps:
        print(
            f"Capping warmup_steps from {args.warmup_steps} to {warmup_steps} "
            f"for max_steps={args.max_steps}",
            flush=True,
        )

    args.resume = _resolve_resume_checkpoint(args.resume, args.output_dir)
    if args.resume:
        print(f"Resuming from: {args.resume}", flush=True)
    else:
        print("Starting clean Stage 2 LoRA run (no checkpoint resume).", flush=True)

    train_path = args.data_dir / "train.jsonl"
    test_path = args.data_dir / "test.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run prepare_brick_dataset.py first.")
        sys.exit(1)

    # Print TRL/transformers versions for debugging
    import trl, transformers
    print(f"TRL version: {trl.__version__}", flush=True)
    print(f"Transformers version: {transformers.__version__}", flush=True)

    print(f"Loading tokenizer: {BRICK_MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BRICK_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...", flush=True)
    ds = load_dataset("json", data_files={
        "train": str(train_path),
        "test": str(test_path),
    })

    # NOTE: HF Trainer shuffles with RandomSampler / DistributedSampler every
    # epoch, which destroys any prefix ordering — so we no longer pre-sort
    # the dataset. ``apply_curriculum_ordering`` is still exported for callers
    # that build their own DataLoader with ``shuffle=False``.
    train_ds = _select_dataset_subset(
        ds["train"],
        args.train_samples,
        seed=args.seed,
        name="train",
        include_tail=256 if args.train_samples is not None else 0,
    )
    eval_ds = _select_dataset_subset(
        ds["test"], args.eval_samples, seed=args.seed + 1, name="eval"
    )
    if not args.no_tokenized_cache:
        train_include_tail = 256 if args.train_samples is not None else 0
        cache_metadata = _tokenized_cache_metadata(
            train_path=train_path,
            test_path=test_path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            train_include_tail=train_include_tail,
            seed=args.seed,
        )
        train_ds, eval_ds = _load_or_create_tokenized_datasets(
            train_ds=train_ds,
            eval_ds=eval_ds,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            cache_root=_tokenized_cache_root(args.data_dir, args.tokenized_cache_dir),
            metadata=cache_metadata,
            rebuild=args.rebuild_tokenized_cache,
        )
    else:
        print("Tokenized dataset cache disabled; SFTTrainer will tokenize datasets.", flush=True)

    print(f"Loading model: {BRICK_MODEL_NAME}", flush=True)
    model_kwargs = {
        "torch_dtype": _resolve_torch_dtype(args.torch_dtype),
        "trust_remote_code": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        BRICK_MODEL_NAME,
        **model_kwargs,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # PEFT warns that DoRA + PiSSA is not a supported combination: PiSSA
    # factorises the base weights via SVD and stores the residual back into
    # the base, and DoRA then decomposes magnitude/direction of those
    # residual weights — the stacked init produces unstable training.
    # Stick with standard Kaiming init plus DoRA + rsLoRA, matching Stage 1.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_parse_lora_target_modules(args.lora_target_modules),
        task_type="CAUSAL_LM",
        use_dora=not args.no_dora,
        use_rslora=not args.no_rslora,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    gradient_checkpointing = not args.no_gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        print("Gradient checkpointing disabled; expect higher VRAM use.", flush=True)

    # ── Structure-aware loss weights ─────────────────────────────────
    structure_weights = BrickStructureWeights(tokenizer)

    output_dir = str(args.output_dir)

    # ── Build training args (version-adaptive) ────────────────────────
    base_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        bf16=USE_BF16,
        fp16=not USE_BF16 and torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none" if args.no_wandb else "wandb",
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise SystemExit("--max-steps must be positive")
        base_kwargs["max_steps"] = args.max_steps
    if gradient_checkpointing:
        base_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # Decide which config class to use
    ConfigClass = SFTConfig if SFTConfig is not None else TrainingArguments
    config_params = _inspect_params(ConfigClass)

    config_kwargs = dict(base_kwargs)
    optional_config_kwargs = {
        "prediction_loss_only": True,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "torch_empty_cache_steps": 10,
        "group_by_length": args.group_by_length,
        "length_column_name": "length",
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
    }
    for key, value in optional_config_kwargs.items():
        if key in config_params:
            config_kwargs[key] = value
    if "max_seq_length" in config_params:
        config_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in config_params:
        config_kwargs["max_length"] = args.max_seq_length
    if "dataset_text_field" in config_params:
        config_kwargs["dataset_text_field"] = None

    print(f"Using config class: {ConfigClass.__name__}", flush=True)
    training_args = ConfigClass(**config_kwargs)
    for key, value in optional_config_kwargs.items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)

    # ── Build custom trainer with structure-aware loss + curriculum ───
    class BrickTrainer(SFTTrainer):
        """SFTTrainer with structure-aware loss and chunked cross-entropy."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Shift labels for causal LM (predict next token)
            shift_labels = labels[..., 1:].contiguous()
            seq_len = shift_labels.size(1)

            # Apply structure-aware weights
            weights = structure_weights.get_weights(shift_labels)

            # Compute CE in token chunks so eval/train do not allocate a giant
            # [batch * sequence, vocab] temporary on top of the model logits.
            vocab_size = logits.size(-1)
            loss_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
            weight_sum = weights.sum().to(device=logits.device, dtype=torch.float32).clamp(min=1.0)
            for start in range(0, seq_len, args.loss_chunk_tokens):
                end = min(start + args.loss_chunk_tokens, seq_len)
                chunk_logits = logits[:, start:end, :].reshape(-1, vocab_size)
                chunk_labels = shift_labels[:, start:end].reshape(-1)
                chunk_weights = weights[:, start:end].reshape(-1).to(
                    device=logits.device,
                    dtype=torch.float32,
                )
                chunk_loss = F.cross_entropy(
                    chunk_logits,
                    chunk_labels,
                    reduction="none",
                    ignore_index=-100,
                ).float()
                loss_sum = loss_sum + (chunk_loss * chunk_weights).sum()

            weighted_loss = loss_sum / weight_sum

            return (weighted_loss, outputs) if return_outputs else weighted_loss

    # ── Build trainer kwargs (version-adaptive) ──────────────────────
    trainer_params = _inspect_params(SFTTrainer)
    print(f"SFTTrainer accepts: processing_class={'processing_class' in trainer_params}, "
          f"tokenizer={'tokenizer' in trainer_params}, "
          f"max_seq_length={'max_seq_length' in trainer_params}", flush=True)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[
            MemoryCleanupCallback(),
        ],
    )

    # Pass tokenizer with the right param name
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Pass max_seq_length to trainer if config didn't accept it
    if "max_seq_length" in trainer_params and "max_seq_length" not in config_kwargs:
        trainer_kwargs["max_seq_length"] = args.max_seq_length

    trainer = BrickTrainer(**trainer_kwargs)

    print("Starting training...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
