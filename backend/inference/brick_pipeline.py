"""Stage 2 inference: text -> colored brick sequence with rejection + rollback.

Also exposes the top-level factory get_brick_pipeline() used by the API routes,
and a MockBrickPipeline for dev mode (LEGOGEN_DEV=1).
"""

import random
import re
import time
from pathlib import Path
from typing import Optional

from backend.brick.constants import BRICK_SHAPES, WORLD_DIM
from backend.brick.parser import Brick, parse_brick, parse_brick_sequence, serialize_brick
from backend.brick.occupancy import VoxelGrid
from backend.brick.stability import is_stable, find_first_unstable
from backend.config import BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, LEGOGEN_DEV
from backend.inference.best_of_n import rank_candidates, cluster_and_pick

_BRICK_RE = re.compile(r"(\d+)x(\d+) \((\d+),(\d+),(\d+)\) #([0-9A-Fa-f]{6})")


def _build_grammar_pattern() -> str:
    """Regex matching one brick line, restricted to allowed dims."""
    dims_alt = "|".join(sorted({f"{h}x{w}" for h, w in BRICK_SHAPES}))
    return rf"({dims_alt}) \(\d{{1,2}},\d{{1,2}},\d{{1,2}}\) #[0-9A-Fa-f]{{6}}\n"


BRICK_PATTERN = _build_grammar_pattern()

MAX_BRICKS = 500
MAX_REJECTIONS = 500
MAX_ROLLBACKS = 100
BASE_TEMPERATURE = 0.6

SYSTEM_PROMPT = "You are a LEGO master builder."
USER_TEMPLATE = (
    "Create a colored LEGO model. Format: <dims> (<x>,<y>,<z>) <#hex>.\n"
    "Allowed dims: 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n"
    "All bricks are 1 unit tall.\n\n### Input:\n{caption}"
)


def _build_logits_processor(tokenizer, pattern: str):
    """Instantiate an outlines RegexLogitsProcessor, or None if outlines is absent."""
    try:
        from outlines.processors import RegexLogitsProcessor
        from outlines.models.transformers import TransformerTokenizer
    except ImportError:
        return None
    return RegexLogitsProcessor(pattern, TransformerTokenizer(tokenizer))


class BrickPipeline:
    """Qwen3.5-4B + brick LoRA. Generates structurally-stable brick sequences.

    Three stacked correctness mechanisms:
      1. Grammar-constrained decoding (regex logits processor) — parse-proof.
      2. Voxel rejection — collision / bounds check against VoxelGrid.
      3. Physics rollback — LP-based stability check via backend/brick/stability.py.
    """

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            BRICK_MODEL_NAME, trust_remote_code=True
        )

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

        self.logits_processor = _build_logits_processor(self.tokenizer, BRICK_PATTERN)

    def generate(self, caption: str, on_progress=None) -> dict:
        """Generate a brick structure for *caption*.

        If ``on_progress`` is provided, it is called with events of the form:
          {"type": "brick", "count": int}
          {"type": "rollback", "count": int}
        """
        import torch

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
        brick_token_lengths: list[int] = []
        base_input_len = input_ids.shape[1]
        grid = VoxelGrid()
        total_rejections = 0
        total_rollbacks = 0

        for _ in range(MAX_ROLLBACKS):
            while len(bricks) < MAX_BRICKS:
                brick, rej = self._generate_one_brick(input_ids, grid)
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
                if on_progress is not None:
                    on_progress({"type": "brick", "count": len(bricks)})

            if is_stable(bricks):
                break

            idx = find_first_unstable(bricks)
            if idx <= 0:
                break

            bricks = bricks[:idx]
            truncated_token_lengths = brick_token_lengths[:idx]
            brick_token_lengths = truncated_token_lengths
            tokens_to_keep = base_input_len + sum(truncated_token_lengths)
            input_ids = input_ids[:, :tokens_to_keep]

            grid.clear()
            for b in bricks:
                grid.place(b)
            total_rollbacks += 1
            if on_progress is not None:
                on_progress({"type": "rollback", "count": total_rollbacks})

        return {
            "bricks": "\n".join(serialize_brick(b) for b in bricks),
            "brick_count": len(bricks),
            "stable": is_stable(bricks),
            "metadata": {
                "model_version": "qwen35-4b-brick-v1",
                "generation_time_ms": int((time.time() - t0) * 1000),
                "rejections": total_rejections,
                "rollbacks": total_rollbacks,
            },
        }

    def generate_best_of_n(self, caption: str, n: int = 16, strategy: str = "cluster", on_progress=None) -> dict:
        """Run generate() n times and return the chosen sample.

        strategy="rank"    -> rank_candidates, return top
        strategy="cluster" -> cluster_and_pick on stable set (falls back to rank)
        """
        candidates: list[dict] = []
        for i in range(n):
            sample = self.generate(caption, on_progress=None)
            sample["bricks_parsed"] = parse_brick_sequence(sample["bricks"])
            candidates.append(sample)
            if on_progress is not None:
                on_progress({"type": "sample", "index": i + 1, "of": n, "stable": sample["stable"]})

        # Ranker/clusterer expect "bricks" to be a list of Brick for features.
        for c in candidates:
            c["bricks"] = c["bricks_parsed"]
        if strategy == "rank":
            picked = rank_candidates(candidates)[0]
        else:
            picked = cluster_and_pick(candidates, k=min(3, n), seed=0)
        picked_index = candidates.index(picked)

        # Restore the serialized "bricks" string on the returned dict.
        picked["bricks"] = "\n".join(serialize_brick(b) for b in picked["bricks"])
        picked.pop("bricks_parsed", None)
        picked.setdefault("metadata", {})
        picked["metadata"]["n"] = n
        picked["metadata"]["picked_index"] = picked_index
        picked["metadata"]["stable_rate"] = sum(1 for c in candidates if c["stable"]) / n
        return picked

    def generate_from_image(self, image, on_progress=None) -> dict:
        """Two-stage: image → Stage 1 caption → brick sequence."""
        caption = _get_stage1_pipeline().describe(image)
        if on_progress is not None:
            on_progress({"type": "caption", "caption": caption})
        result = self.generate(caption, on_progress=on_progress)
        result["caption"] = caption
        return result

    def _generate_one_brick(self, input_ids, grid: VoxelGrid) -> tuple[Optional[Brick], int]:
        import torch

        gen_kwargs = dict(
            max_new_tokens=30,
            temperature=BASE_TEMPERATURE,
            do_sample=True,
            top_k=20,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.logits_processor is not None:
            gen_kwargs["logits_processor"] = [self.logits_processor]

        for attempt in range(MAX_REJECTIONS):
            with torch.no_grad():
                out = self.model.generate(input_ids, **gen_kwargs)
            text = self.tokenizer.decode(
                out[0, input_ids.shape[1]:], skip_special_tokens=False
            )
            if self.tokenizer.eos_token in text or not text.strip():
                return None, attempt
            first_line = text.strip().split("\n")[0].strip()
            try:
                brick = parse_brick(first_line)
            except ValueError:
                # Should not occur under the outlines grammar processor, but
                # the optional-import fallback at _build_logits_processor
                # returns None when outlines is missing, and at that point a
                # malformed line becomes possible. Treat it as a rejection
                # rather than propagating a 500.
                continue
            if not grid.can_place(brick):
                continue
            return brick, attempt
        return None, MAX_REJECTIONS


class MockBrickPipeline:
    """Deterministic-ish mock used when LEGOGEN_DEV=1.

    Emits a small red house (~12 bricks) so the frontend has renderable data.
    """

    _HOUSE_BRICKS = [
        # Base plate layer (z=0) — 4x2 footprint
        ("2x4", 0, 0, 0, "#237841"),
        ("2x4", 2, 0, 0, "#237841"),
        # Lower walls (z=1)
        ("2x4", 0, 0, 1, "#C91A09"),
        ("2x4", 2, 0, 1, "#C91A09"),
        ("1x2", 0, 2, 1, "#C91A09"),
        ("1x2", 3, 2, 1, "#C91A09"),
        # Upper walls (z=2)
        ("2x4", 0, 0, 2, "#FFFFFF"),
        ("2x4", 2, 0, 2, "#FFFFFF"),
        # Roof (z=3, z=4)
        ("2x4", 0, 0, 3, "#FE8A18"),
        ("2x4", 2, 0, 3, "#FE8A18"),
        ("2x2", 1, 1, 4, "#720E0F"),
        ("2x2", 2, 1, 4, "#720E0F"),
    ]

    def generate(self, caption: str, on_progress=None) -> dict:
        t0 = time.time()
        lines = []
        for i, (dims, x, y, z, color) in enumerate(self._HOUSE_BRICKS):
            lines.append(f"{dims} ({x},{y},{z}) {color}")
            if on_progress is not None:
                on_progress({"type": "brick", "count": i + 1})
        return {
            "bricks": "\n".join(lines),
            "brick_count": len(lines),
            "stable": True,
            "metadata": {
                "model_version": "mock-brick-v1",
                "generation_time_ms": max(1, int((time.time() - t0) * 1000)),
                "rejections": 0,
                "rollbacks": 0,
            },
        }

    def generate_best_of_n(self, caption: str, n: int = 16, strategy: str = "rank", on_progress=None) -> dict:
        result = self.generate(caption, on_progress=on_progress)
        result.setdefault("metadata", {})
        result["metadata"]["n"] = n
        result["metadata"]["picked_index"] = 0
        result["metadata"]["stable_rate"] = 1.0
        return result

    def generate_from_image(self, image, on_progress=None) -> dict:
        caption = "a small red house with a dark red tiled roof"
        if on_progress is not None:
            on_progress({"type": "caption", "caption": caption})
        result = self.generate(caption, on_progress=on_progress)
        result["caption"] = caption
        return result


# ── Singleton factories ──────────────────────────────────────────────

_brick_instance = None
_stage1_instance = None


def get_brick_pipeline():
    global _brick_instance
    if _brick_instance is None:
        if LEGOGEN_DEV:
            _brick_instance = MockBrickPipeline()
        else:
            _brick_instance = BrickPipeline()
    return _brick_instance


def _get_stage1_pipeline():
    global _stage1_instance
    if _stage1_instance is None:
        if LEGOGEN_DEV:
            _stage1_instance = _MockStage1()
        else:
            from backend.inference.stage1_pipeline import Stage1Pipeline
            _stage1_instance = Stage1Pipeline()
    return _stage1_instance


class _MockStage1:
    def describe(self, image) -> str:
        return "a small red house with a dark red tiled roof"
