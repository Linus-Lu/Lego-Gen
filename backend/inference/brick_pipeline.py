"""Stage 2 inference: text -> colored brick sequence with rejection + rollback.

Adopts several techniques from BrickGPT:
  - Few-shot prompting with demonstration examples
  - Per-brick rejection reason tracking (categorised counter)
  - Rejected-brick deduplication (set-based, temperature only rises on repeats)
  - Incremental connectivity checking via ReliabilityScorer
  - Optional constrained decoding via logit masking
  - KV-cache reuse across rejection attempts for efficiency
"""

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from backend.brick.constants import BRICK_SHAPES, WORLD_DIM
from backend.brick.parser import Brick, parse_brick, serialize_brick
from backend.brick.occupancy import VoxelGrid
from backend.brick.stability import is_stable, find_first_unstable, is_brick_connected
from backend.brick.reliability import ReliabilityScorer
from backend.brick.decoder import build_brick_logits_processor
from backend.config import BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, RELIABILITY_SCORE_THRESHOLD

_BRICK_RE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})")

MAX_BRICKS = 500
MAX_REJECTIONS = 500
MAX_ROLLBACKS = 100
BASE_TEMPERATURE = 0.6
TEMP_INCREMENT = 0.01
MAX_TEMPERATURE = 2.0

SYSTEM_PROMPT = "You are a LEGO master builder."

# ── Few-shot examples ────────────────────────────────────────────────
_FEW_SHOT_PATH = Path(__file__).resolve().parent.parent / "brick" / "few_shot_examples.json"


def _load_few_shot_examples() -> list[dict]:
    if _FEW_SHOT_PATH.exists():
        with open(_FEW_SHOT_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return []


_FEW_SHOT_EXAMPLES = _load_few_shot_examples()


def _build_few_shot_block() -> str:
    """Format the few-shot examples as a prompt block."""
    if not _FEW_SHOT_EXAMPLES:
        return ""
    parts = ["Here are some example LEGO models:\n"]
    for ex in _FEW_SHOT_EXAMPLES:
        parts.append(f"### Input:\n{ex['caption']}\n### Output:\n{ex['bricks']}\n")
    parts.append(
        "Do NOT copy the examples. Create your own LEGO model for the "
        "following input.\n"
    )
    return "\n".join(parts)


_FEW_SHOT_BLOCK = _build_few_shot_block()

USER_TEMPLATE = (
    "Create a colored LEGO model. Format: <dims> (<x>,<y>,<z>) <#hex>.\n"
    "Allowed dims: 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n"
    "All bricks are 1 unit tall.\n\n"
    "{few_shot}"
    "### Input:\n{caption}"
)

# ── Rejection reason labels ──────────────────────────────────────────
_REJ_FORMAT = "ill_formatted"
_REJ_COLLISION = "collision"
_REJ_DISCONNECTED = "disconnected"
_REJ_ALREADY = "already_rejected"


class BrickPipeline:
    """Generate LEGO brick structures from text captions using a fine-tuned LLM.

    Uses rejection sampling (invalid bricks are discarded and regenerated with
    increasing temperature) and physics rollback (when an unstable brick is
    detected, the sequence is truncated to the last stable prefix and generation
    resumes from there).
    """

    def __init__(self, device: str = "cuda") -> None:
        from transformers import BitsAndBytesConfig

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            BRICK_MODEL_NAME, trust_remote_code=True
        )

        # Use 4-bit quantization to reduce VRAM (~2GB instead of ~8GB for bf16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BRICK_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        ckpt = Path(BRICK_CHECKPOINT_DIR)
        self.model = (
            PeftModel.from_pretrained(base, str(ckpt)) if ckpt.exists() else base
        )
        self.model.eval()

        # Try to build a constrained-decoding logits processor.  If the
        # tokeniser is incompatible (some strings need >1 token) we fall
        # back to regex-based post-validation.
        self._logits_processor = build_brick_logits_processor(
            self.tokenizer, eos_token_id=self.tokenizer.eos_token_id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, caption: str) -> dict:
        """Generate a brick structure for *caption*.

        Returns a dict with keys:
        - bricks: newline-separated brick text
        - brick_count: number of bricks placed
        - stable: whether the final structure is stable
        - metadata: generation statistics (including reliability scores)
        """
        t0 = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    caption=caption, few_shot=_FEW_SHOT_BLOCK
                ),
            },
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )

        bricks: list[Brick] = []
        brick_token_lengths: list[int] = []
        base_input_len = input_ids.shape[1]
        grid = VoxelGrid()
        scorer = ReliabilityScorer(grid)
        rejection_reasons: Counter = Counter()
        total_rollbacks = 0

        # Pre-fill KV cache for the prompt so rejection attempts inside
        # _generate_one_brick only process new tokens.
        kv_cache = self._prefill_cache(input_ids)

        for _ in range(MAX_ROLLBACKS):
            while len(bricks) < MAX_BRICKS:
                result = self._generate_one_brick(
                    input_ids, grid, scorer, kv_cache
                )
                rejection_reasons.update(result["reasons"])
                brick = result["brick"]
                if brick is None:
                    break
                # Score the brick *before* placing in grid (support_ratio
                # reads the pre-placement occupancy).
                scorer.add_brick(brick)
                bricks.append(brick)
                grid.place(brick)
                brick_ids = self.tokenizer.encode(
                    serialize_brick(brick) + "\n", add_special_tokens=False
                )
                brick_token_lengths.append(len(brick_ids))
                brick_id_tensor = torch.tensor(
                    [brick_ids], device=self.device
                )
                input_ids = torch.cat([input_ids, brick_id_tensor], dim=1)

                # Extend the KV cache with the accepted brick's tokens.
                kv_cache = self._extend_cache(kv_cache, brick_id_tensor)

            # Safety-net: full connectivity check (should rarely fail now
            # that every brick is checked incrementally).
            if is_stable(bricks):
                break

            idx = find_first_unstable(bricks)
            if idx <= 0:
                break

            # Rollback
            bricks = bricks[:idx]
            truncated_token_lengths = brick_token_lengths[:idx]
            brick_token_lengths = truncated_token_lengths
            tokens_to_keep = base_input_len + sum(truncated_token_lengths)
            input_ids = input_ids[:, :tokens_to_keep]

            grid.clear()
            for b in bricks:
                grid.place(b)
            scorer.remove_last(len(scorer.scores) - idx)
            total_rollbacks += 1

            # Rebuild KV cache after rollback.
            kv_cache = self._prefill_cache(input_ids)

        reliability_scores = [s.score for s in scorer.scores]
        return {
            "bricks": "\n".join(serialize_brick(b) for b in bricks),
            "brick_count": len(bricks),
            "stable": is_stable(bricks),
            "metadata": {
                "model_version": "qwen35-4b-brick-v1",
                "generation_time_ms": int((time.time() - t0) * 1000),
                "rejections": rejection_reasons.total(),
                "rejection_reasons": dict(rejection_reasons),
                "rollbacks": total_rollbacks,
                "avg_reliability_score": scorer.aggregate_score(),
                "min_reliability_score": scorer.min_score(),
                "reliability_scores": reliability_scores,
            },
        }

    # ------------------------------------------------------------------
    # Per-brick generation with rejection sampling
    # ------------------------------------------------------------------

    def _generate_one_brick(
        self,
        input_ids: torch.Tensor,
        grid: VoxelGrid,
        scorer: ReliabilityScorer,
        kv_cache: object | None,
    ) -> dict:
        """Generate one valid brick via rejection sampling.

        Each candidate must pass four checks:
        1. **Format** — matches ``HxW (x,y,z) #RRGGBB`` regex.
        2. **Collision** — ``grid.can_place`` confirms no overlap.
        3. **Connectivity** — brick connects to the ground-reachable set.
        4. **Deduplication** — not in the already-rejected set.

        Returns ``{"brick": Brick | None, "reasons": Counter}``.
        """
        temp = BASE_TEMPERATURE
        rejected_set: set[str] = set()
        reasons: Counter = Counter()

        # If we have a constrained-decoding processor, reset it.
        logits_proc = self._logits_processor
        gen_kwargs: dict = {}
        if logits_proc is not None:
            logits_proc.reset()
            gen_kwargs["logits_processor"] = [logits_proc]

        for attempt in range(MAX_REJECTIONS):
            with torch.no_grad():
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=30,
                    temperature=temp,
                    do_sample=True,
                    top_k=20,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    past_key_values=kv_cache,
                    use_cache=True,
                    **gen_kwargs,
                )
            text = self.tokenizer.decode(
                out[0, input_ids.shape[1] :], skip_special_tokens=False
            )
            if self.tokenizer.eos_token in text or not text.strip():
                return {"brick": None, "reasons": reasons}

            first_line = text.strip().split("\n")[0].strip()

            # ── Deduplication check ───────────────────────────────────
            if first_line in rejected_set:
                reasons[_REJ_ALREADY] += 1
                temp = min(temp + TEMP_INCREMENT, MAX_TEMPERATURE)
                continue

            # ── Format check ──────────────────────────────────────────
            m = _BRICK_RE.fullmatch(first_line)
            if not m:
                reasons[_REJ_FORMAT] += 1
                rejected_set.add(first_line)
                continue  # no temp increase for format errors

            brick = parse_brick(first_line)

            # ── Collision check ───────────────────────────────────────
            if not grid.can_place(brick):
                reasons[_REJ_COLLISION] += 1
                rejected_set.add(first_line)
                continue  # no temp increase for collisions

            # ── Connectivity check ────────────────────────────────────
            if not is_brick_connected(
                brick, scorer._bricks, scorer._ground_set
            ):
                reasons[_REJ_DISCONNECTED] += 1
                rejected_set.add(first_line)
                continue  # no temp increase for disconnected

            return {"brick": brick, "reasons": reasons}

        return {"brick": None, "reasons": reasons}

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def _prefill_cache(self, input_ids: torch.Tensor) -> object | None:
        """Run a forward pass over *input_ids* and return the KV cache.

        The cache allows subsequent ``generate()`` calls to skip reprocessing
        the prompt + accepted bricks, processing only new candidate tokens.
        """
        try:
            with torch.no_grad():
                out = self.model(input_ids, use_cache=True)
            return out.past_key_values
        except Exception:
            return None

    def _extend_cache(
        self, kv_cache: object | None, new_ids: torch.Tensor
    ) -> object | None:
        """Extend *kv_cache* with the representations of *new_ids*."""
        if kv_cache is None:
            return None
        try:
            with torch.no_grad():
                out = self.model(new_ids, past_key_values=kv_cache, use_cache=True)
            return out.past_key_values
        except Exception:
            return None
