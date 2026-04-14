"""Two-phase inference pipeline.

Stage 1: Image → text description (Qwen3.5-9B + LoRA)
Stage 2: Text → brick coordinates (Qwen3.5-4B + LoRA)

Integrates three caching layers (when CACHE_ENABLED):
  1. KV prefix cache -- reuses pre-computed KV states for Stage 1 system prompt
  2. Response cache  -- LRU+TTL cache for identical inputs
  3. Tokenization cache -- cached chat-template strings for static prompts
"""

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    UNIFIED_MODEL_NAME,
    UNIFIED_CHECKPOINT_DIR,
    STAGE1_CHECKPOINT_DIR,
    STAGE1_SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    LEGOGEN_DEV,
    CACHE_ENABLED,
    CACHE_KV_PREFIX_ENABLED,
    CACHE_RESPONSE_ENABLED,
    CACHE_RESPONSE_FOR_SAMPLING,
    CACHE_TOKENIZATION_ENABLED,
    CACHE_RESPONSE_MAX_SIZE,
    CACHE_RESPONSE_TTL_SECONDS,
)

# Response caching is only effective when results are deterministic.
# With do_sample=True and temperature>0, skip response cache unless explicitly opted in.
_RESPONSE_CACHE_ACTIVE = CACHE_RESPONSE_ENABLED and (
    TEMPERATURE == 0 or not TOP_P or CACHE_RESPONSE_FOR_SAMPLING
)


# ── Singletons ────────────────────────────────────────────────────────

_pipeline_instance = None


def get_pipeline():
    """Get or create the pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        if LEGOGEN_DEV:
            _pipeline_instance = MockPipeline()
        else:
            _pipeline_instance = TwoStagePipeline()
    return _pipeline_instance


def get_planner_pipeline():
    """Returns the same pipeline instance."""
    return get_pipeline()


# ── Mock pipeline for frontend development ─────────────────────────────

class MockPipeline:
    """Returns a realistic hardcoded brick response for dev/testing."""

    has_stage1 = True

    def generate_brick_build(self, caption: str) -> dict:
        """Mock text-to-bricks pipeline for dev mode."""
        from backend.brick.parser import serialize_brick, Brick

        bricks = [
            Brick(h=2, w=4, x=0, y=0, z=0, color="C91A09"),
            Brick(h=2, w=4, x=2, y=0, z=0, color="0055BF"),
            Brick(h=2, w=4, x=4, y=0, z=0, color="C91A09"),
            Brick(h=2, w=4, x=1, y=0, z=1, color="FFFFFF"),
            Brick(h=2, w=4, x=3, y=0, z=1, color="FFFFFF"),
            Brick(h=2, w=4, x=0, y=0, z=2, color="FEC401"),
            Brick(h=2, w=4, x=2, y=0, z=2, color="FEC401"),
            Brick(h=2, w=4, x=4, y=0, z=2, color="FEC401"),
        ]

        return {
            "bricks": "\n".join(serialize_brick(b) for b in bricks),
            "caption": caption or "A small colorful house",
            "brick_count": len(bricks),
            "stable": True,
            "metadata": {
                "model_version": "mock-dev-v1",
                "generation_time_ms": 42,
                "rejections": 0,
                "rollbacks": 0,
            },
        }

    def generate_brick_build_from_image(self, image) -> dict:
        """Mock image-to-bricks pipeline for dev mode."""
        result = self.generate_brick_build("A small colorful house")
        result["caption"] = "A small colorful house"
        return result

    def describe_image_stage1(self, image) -> str:
        """Mock Stage 1: image → description."""
        return "A small colorful house with red walls, white trim, and a yellow roof."


# ── Two-Stage Pipeline ─────────────────────────────────────────────────

class TwoStagePipeline:
    """Two-phase pipeline: image → text description → brick coordinates.

    Stage 1: Uses Qwen VL model with LoRA adapter for image → text
    Stage 2: Uses Qwen 4B model for text → brick coordinates

    Integrates three caching layers (when CACHE_ENABLED):
      1. KV prefix cache -- reuses pre-computed KV states for Stage 1 system prompt
      2. Response cache  -- LRU+TTL cache for identical inputs
      3. Tokenization cache -- cached chat-template strings for static prompts
    """

    def __init__(
        self,
        adapter_path: str | Path | None = None,
        model_name: str = UNIFIED_MODEL_NAME,
    ):
        import torch
        from backend.models.unified_model import LegoUnifiedModel

        if adapter_path is None:
            adapter_path = UNIFIED_CHECKPOINT_DIR

        adapter_path = Path(adapter_path)
        if adapter_path.exists():
            load_adapter = str(adapter_path)
        else:
            print(f"Checkpoint not found at {adapter_path}, using base model")
            load_adapter = None

        self.wrapper = LegoUnifiedModel(
            model_name=model_name,
            load_adapter=load_adapter,
            is_trainable=False,
        )
        self.model = self.wrapper.get_model()
        self.processor = self.wrapper.get_processor()
        self.model.eval()

        # Try to load Stage 1 adapter for two-stage pipeline
        stage1_path = Path(STAGE1_CHECKPOINT_DIR)
        self.has_stage1 = False
        if stage1_path.exists() and (stage1_path / "adapter_config.json").exists():
            self.has_stage1 = self.wrapper.load_named_adapter("stage1", stage1_path)
            if self.has_stage1:
                self.wrapper.set_adapter("default")
                print("Two-stage pipeline enabled (Stage 1 + Stage 2)")

        # Initialize caching layers
        self._init_caches()

    def _init_caches(self):
        """Initialize and warm up all cache layers."""
        from backend.inference.cache import KVPrefixCache, ResponseCache, TokenizationCache

        self.kv_cache = KVPrefixCache()
        self.response_cache = ResponseCache(
            max_size=CACHE_RESPONSE_MAX_SIZE,
            ttl_seconds=CACHE_RESPONSE_TTL_SECONDS,
        )
        self.tokenization_cache = TokenizationCache()

        if not CACHE_ENABLED:
            print("[cache] Caching disabled via LEGOGEN_CACHE_ENABLED=0")
            return

        self._warmup_caches()

    def _warmup_caches(self):
        """Pre-compute KV prefixes and tokenization caches at startup."""
        start = time.time()

        # Layer 1: KV prefix cache for Stage 1 system prompt
        if CACHE_KV_PREFIX_ENABLED and self.has_stage1:
            try:
                self.wrapper.set_adapter("stage1")
                self.kv_cache.warmup_stage1(self.model, self.processor, STAGE1_SYSTEM_PROMPT)
                self.wrapper.set_adapter("default")
            except Exception as e:
                print(f"[cache] Stage 1 KV prefix warmup failed (non-fatal): {e}")
                self.wrapper.set_adapter("default")

        # Layer 3: Tokenization cache for Stage 1 chat template
        if CACHE_TOKENIZATION_ENABLED:
            try:
                self.tokenization_cache.get_or_compute(
                    "stage1_template",
                    lambda: self.processor.apply_chat_template(
                        [
                            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": "Describe this object for LEGO building."},
                                ],
                            },
                        ],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                )
            except Exception as e:
                print(f"[cache] Tokenization warmup failed (non-fatal): {e}")

        elapsed = int((time.time() - start) * 1000)
        print(f"[cache] Warmup complete in {elapsed}ms")

    def describe_image_stage1(self, image) -> str:
        """Stage 1: Generate a structural description from an image.

        Uses KV prefix cache for the Stage 1 system prompt when available.
        """
        import torch
        from backend.models.tokenizer import strip_thinking_blocks

        if not self.has_stage1:
            print("WARNING: Stage 1 adapter not loaded — using default adapter for image description")

        if self.has_stage1:
            self.wrapper.set_adapter("stage1")

        with torch.inference_mode():
            # Layer 3: Use cached template string or compute fresh
            if CACHE_ENABLED and CACHE_TOKENIZATION_ENABLED:
                text = self.tokenization_cache.get("stage1_template")
            else:
                text = None

            if text is None:
                messages = [
                    {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe this object for LEGO building."},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )

            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt",
            ).to(self.model.device)

            generate_kwargs = dict(
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            # Layer 1: KV prefix cache for Stage 1 system prompt
            if (
                CACHE_ENABLED
                and CACHE_KV_PREFIX_ENABLED
                and self.kv_cache.is_ready
            ):
                kv_clone, prefix_len = self.kv_cache.get_stage1_prefix()
                if kv_clone is not None and inputs["input_ids"].shape[1] > prefix_len:
                    new_input_ids = inputs["input_ids"][:, prefix_len:]
                    full_len = inputs["input_ids"].shape[1]
                    attention_mask = torch.ones(
                        (1, full_len), dtype=torch.long, device=self.model.device
                    )
                    position_ids = torch.arange(
                        prefix_len, full_len, dtype=torch.long, device=self.model.device
                    ).unsqueeze(0)

                    generate_kwargs["past_key_values"] = kv_clone
                    generate_kwargs["attention_mask"] = attention_mask

                    extra_keys = {}
                    for k in ("pixel_values", "image_grid_thw"):
                        if k in inputs:
                            extra_keys[k] = inputs[k]

                    outputs = self.model.generate(
                        input_ids=new_input_ids,
                        position_ids=position_ids,
                        **extra_keys,
                        **generate_kwargs,
                    )
                    generated_ids = outputs[0][new_input_ids.shape[1]:]
                else:
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            else:
                outputs = self.model.generate(**inputs, **generate_kwargs)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

            raw = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        raw = strip_thinking_blocks(raw)

        # Switch back to default adapter
        if self.has_stage1:
            self.wrapper.set_adapter("default")

        return raw.strip()

    def generate_brick_build(self, caption: str) -> dict:
        """Stage 2: Generate brick coordinates from text using Qwen 4B.

        Checks response cache first (Layer 2) when caching is active.
        """
        from backend.inference.cache import ResponseCache

        # Layer 2: Check response cache
        cache_key = None
        if CACHE_ENABLED and _RESPONSE_CACHE_ACTIVE:
            cache_key = f"brick:{ResponseCache.make_key_text(caption)}"
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                return {**cached, "metadata": {**cached["metadata"], "cached": True}}

        from backend.inference.brick_pipeline import BrickPipeline
        if not hasattr(self, '_brick_pipeline'):
            self._brick_pipeline = BrickPipeline(device=str(self.model.device))
        result = self._brick_pipeline.generate(caption)

        # Layer 2: Store in response cache
        if CACHE_ENABLED and _RESPONSE_CACHE_ACTIVE and cache_key:
            self.response_cache.put(cache_key, result)

        return result

    def generate_brick_build_from_image(self, image) -> dict:
        """Full two-stage: image → Stage 1 caption → Stage 2 bricks."""
        caption = self.describe_image_stage1(image)
        result = self.generate_brick_build(caption)
        result["caption"] = caption
        return result

    def cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "kv_prefix_ready": self.kv_cache.is_ready,
            "response_cache": self.response_cache.stats(),
            "response_cache_active": _RESPONSE_CACHE_ACTIVE,
        }
