"""Stage 2 inference: text -> colored brick sequence with rejection + rollback."""

import re
import time
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
from backend.config import BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, RELIABILITY_SCORE_THRESHOLD

_BRICK_RE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})")

MAX_BRICKS = 500
MAX_REJECTIONS = 500
MAX_ROLLBACKS = 100
BASE_TEMPERATURE = 0.6
TEMP_INCREMENT = 0.01
MAX_TEMPERATURE = 2.0

SYSTEM_PROMPT = "You are a LEGO master builder."
USER_TEMPLATE = (
    "Create a colored LEGO model. Format: <dims> (<x>,<y>,<z>) <#hex>.\n"
    "Allowed dims: 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n"
    "All bricks are 1 unit tall.\n\n### Input:\n{caption}"
)


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
            {"role": "user", "content": USER_TEMPLATE.format(caption=caption)},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )

        bricks: list[Brick] = []
        # Track the token length of each brick for efficient rollback
        brick_token_lengths: list[int] = []
        base_input_len = input_ids.shape[1]
        grid = VoxelGrid()
        scorer = ReliabilityScorer(grid)
        total_rejections = 0
        total_rollbacks = 0

        for _ in range(MAX_ROLLBACKS):
            while len(bricks) < MAX_BRICKS:
                brick, rej = self._generate_one_brick(input_ids, grid, scorer)
                total_rejections += rej
                if brick is None:
                    break
                # Score the brick *before* placing it in the grid so
                # support_ratio reads the pre-placement occupancy.
                brick_score = scorer.add_brick(brick)
                bricks.append(brick)
                grid.place(brick)
                brick_ids = self.tokenizer.encode(
                    serialize_brick(brick) + "\n", add_special_tokens=False
                )
                brick_token_lengths.append(len(brick_ids))
                input_ids = torch.cat(
                    [input_ids, torch.tensor([brick_ids], device=self.device)], dim=1
                )

            # Safety-net: full connectivity check (should rarely fail now
            # that every brick is checked incrementally).
            if is_stable(bricks):
                break

            idx = find_first_unstable(bricks)
            if idx <= 0:
                break

            # Rollback by slicing input_ids instead of re-tokenizing
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

        reliability_scores = [s.score for s in scorer.scores]
        return {
            "bricks": "\n".join(serialize_brick(b) for b in bricks),
            "brick_count": len(bricks),
            "stable": is_stable(bricks),
            "metadata": {
                "model_version": "qwen35-4b-brick-v1",
                "generation_time_ms": int((time.time() - t0) * 1000),
                "rejections": total_rejections,
                "rollbacks": total_rollbacks,
                "avg_reliability_score": scorer.aggregate_score(),
                "min_reliability_score": scorer.min_score(),
                "reliability_scores": reliability_scores,
            },
        }

    def _generate_one_brick(
        self,
        input_ids: torch.Tensor,
        grid: VoxelGrid,
        scorer: ReliabilityScorer,
    ) -> tuple[Optional[Brick], int]:
        """Generate one valid brick via rejection sampling.

        Each candidate brick must pass three checks before it is accepted:
        1. **Format** — matches the ``HxW (x,y,z) #RRGGBB`` regex.
        2. **Collision** — ``grid.can_place`` confirms no overlap.
        3. **Connectivity** — the brick must connect to the ground-reachable
           set tracked by *scorer* (incremental stability check inspired by
           BrickGPT's per-step reliability verification).

        Returns (Brick or None, rejection_count). Returns None when the model
        emits EOS or empty output, signalling end of generation.
        """
        temp = BASE_TEMPERATURE
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
                )
            text = self.tokenizer.decode(
                out[0, input_ids.shape[1] :], skip_special_tokens=False
            )
            if self.tokenizer.eos_token in text or not text.strip():
                return None, attempt
            first_line = text.strip().split("\n")[0].strip()
            m = _BRICK_RE.fullmatch(first_line)
            if not m:
                temp = min(temp + TEMP_INCREMENT, MAX_TEMPERATURE)
                continue
            brick = parse_brick(first_line)
            if not grid.can_place(brick):
                temp = min(temp + TEMP_INCREMENT, MAX_TEMPERATURE)
                continue
            # Incremental connectivity check: reject floating bricks
            # immediately rather than discovering them post-generation.
            if not is_brick_connected(
                brick, scorer._bricks, scorer._ground_set
            ):
                temp = min(temp + TEMP_INCREMENT, MAX_TEMPERATURE)
                continue
            return brick, attempt
        return None, MAX_REJECTIONS
