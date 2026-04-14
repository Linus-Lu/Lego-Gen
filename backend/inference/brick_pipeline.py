"""Stage 2 inference: text -> colored brick sequence with rejection + rollback."""

import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from backend.brick.parser import Brick, parse_brick, serialize_brick, _BRICK_RE
from backend.brick.occupancy import VoxelGrid
from backend.brick.stability import is_stable, find_first_unstable
from backend.config import BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR

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
        - metadata: generation statistics
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
        past_key_values = None
        total_rejections = 0
        total_rollbacks = 0
        structure_stable = True

        for _ in range(MAX_ROLLBACKS):
            while len(bricks) < MAX_BRICKS:
                brick, rej, past_key_values = self._generate_one_brick(
                    input_ids, grid, past_key_values
                )
                total_rejections += rej
                if brick is None:
                    break
                bricks.append(brick)
                grid.place(brick)
                brick_ids = self.tokenizer.encode(
                    serialize_brick(brick) + "\n", add_special_tokens=False
                )
                brick_token_lengths.append(len(brick_ids))
                input_ids = torch.cat(
                    [input_ids, torch.tensor([brick_ids], device=self.device)], dim=1
                )

            structure_stable = is_stable(bricks)
            if structure_stable:
                break

            idx = find_first_unstable(bricks)
            if idx <= 0:
                break

            # Remove only the dropped bricks from the grid instead of
            # clearing and re-placing all remaining bricks.
            for b in bricks[idx:]:
                grid.remove(b)
            bricks = bricks[:idx]
            brick_token_lengths = brick_token_lengths[:idx]
            tokens_to_keep = base_input_len + sum(brick_token_lengths)
            input_ids = input_ids[:, :tokens_to_keep]

            # Invalidate KV cache after rollback since the sequence changed
            past_key_values = None
            total_rollbacks += 1

        return {
            "bricks": "\n".join(serialize_brick(b) for b in bricks),
            "brick_count": len(bricks),
            "stable": structure_stable,
            "metadata": {
                "model_version": "qwen35-4b-brick-v1",
                "generation_time_ms": int((time.time() - t0) * 1000),
                "rejections": total_rejections,
                "rollbacks": total_rollbacks,
            },
        }

    def _generate_one_brick(
        self,
        input_ids: torch.Tensor,
        grid: VoxelGrid,
        past_key_values=None,
    ) -> tuple[Optional[Brick], int, any]:
        """Generate one valid brick via rejection sampling.

        Returns (Brick or None, rejection_count, past_key_values). Returns
        None when the model emits EOS or empty output, signalling end of
        generation. Passes through the KV cache for reuse on the next call.
        """
        temp = BASE_TEMPERATURE
        for attempt in range(MAX_REJECTIONS):
            with torch.inference_mode():
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
                return None, attempt, None
            first_line = text.strip().split("\n")[0].strip()
            m = _BRICK_RE.fullmatch(first_line)
            if not m:
                temp = min(temp + TEMP_INCREMENT, MAX_TEMPERATURE)
                continue
            brick = parse_brick(first_line)
            if not grid.can_place(brick):
                temp = min(temp + TEMP_INCREMENT, MAX_TEMPERATURE)
                continue
            return brick, attempt, None
        return None, MAX_REJECTIONS, None
