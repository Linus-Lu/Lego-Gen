"""End-to-end inference pipelines: image/text -> JSON description -> build steps."""

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    MODEL_NAME,
    UNIFIED_MODEL_NAME,
    PLANNER_MODEL_NAME,
    CHECKPOINT_DIR,
    PLANNER_CHECKPOINT_DIR,
    UNIFIED_CHECKPOINT_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    LEGOGEN_DEV,
    CACHE_ENABLED,
    CACHE_KV_PREFIX_ENABLED,
    CACHE_RESPONSE_ENABLED,
    CACHE_TOKENIZATION_ENABLED,
    CACHE_RESPONSE_MAX_SIZE,
    CACHE_RESPONSE_TTL_SECONDS,
)


# ── Singletons ────────────────────────────────────────────────────────

_unified_instance = None


def get_pipeline():
    """Get or create the unified pipeline instance (handles both image and text)."""
    global _unified_instance
    if _unified_instance is None:
        if LEGOGEN_DEV:
            _unified_instance = MockPipeline()
        else:
            _unified_instance = UnifiedPipeline()
    return _unified_instance


def get_planner_pipeline():
    """Returns the same unified pipeline instance."""
    return get_pipeline()


# ── Mock pipeline for frontend development ─────────────────────────────

class MockPipeline:
    """Returns a realistic hardcoded LEGO house response for dev/testing."""

    def generate_build_from_text(self, prompt: str) -> dict:
        """Mock text-to-JSON pipeline for dev mode."""
        return self.generate_build(image=None)

    def generate_build(self, image=None, cache_key=None) -> dict:
        from backend.inference.postprocess_manual import json_to_steps
        from backend.inference.stability_checker import StabilityChecker
        from dataclasses import asdict

        description = {
            "set_id": "mock-001",
            "object": "Cozy Family House",
            "category": "City",
            "subcategory": "Residential",
            "complexity": "intermediate",
            "total_parts": 86,
            "dominant_colors": ["Red", "White", "Bright Orange"],
            "dimensions_estimate": {"width": "medium", "height": "medium", "depth": "small"},
            "subassemblies": [
                {
                    "name": "base_plate",
                    "type": "Baseplates",
                    "parts": [
                        {"part_id": "3811", "name": "Baseplate 32x32", "category": "Baseplates", "color": "Green", "color_hex": "#237841", "is_trans": False, "quantity": 1},
                        {"part_id": "3020", "name": "Plate 2x4", "category": "Plates", "color": "Dark Tan", "color_hex": "#958A73", "is_trans": False, "quantity": 4},
                    ],
                    "spatial": {"position": "bottom", "orientation": "flat", "connects_to": ["walls_lower"]},
                },
                {
                    "name": "walls_lower",
                    "type": "Bricks",
                    "parts": [
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 8},
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 6},
                        {"part_id": "3010", "name": "Brick 1x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 4},
                    ],
                    "spatial": {"position": "bottom", "orientation": "upright", "connects_to": ["walls_upper"]},
                },
                {
                    "name": "walls_upper",
                    "type": "Bricks",
                    "parts": [
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Yellow", "color_hex": "#F2CD37", "is_trans": False, "quantity": 6},
                        {"part_id": "3001", "name": "Brick 2x4", "category": "Bricks", "color": "Red", "color_hex": "#C91A09", "is_trans": False, "quantity": 4},
                        {"part_id": "3622", "name": "Brick 1x3", "category": "Bricks", "color": "White", "color_hex": "#FFFFFF", "is_trans": False, "quantity": 4},
                    ],
                    "spatial": {"position": "center", "orientation": "upright", "connects_to": ["windows_and_doors", "roof"]},
                },
                {
                    "name": "windows_and_doors",
                    "type": "Windows and Doors",
                    "parts": [
                        {"part_id": "60594", "name": "Window 1x2x3 Pane", "category": "Windows and Doors", "color": "Trans-Clear", "color_hex": "#FCFCFC", "is_trans": True, "quantity": 4},
                        {"part_id": "60593", "name": "Window 1x2x3 Frame", "category": "Windows and Doors", "color": "Blue", "color_hex": "#0055BF", "is_trans": False, "quantity": 4},
                        {"part_id": "60596", "name": "Door 1x4x6 Frame", "category": "Windows and Doors", "color": "Reddish Brown", "color_hex": "#582A12", "is_trans": False, "quantity": 1},
                        {"part_id": "60616", "name": "Door 1x4x6 Panel", "category": "Windows and Doors", "color": "Dark Azure", "color_hex": "#078BC9", "is_trans": False, "quantity": 1},
                    ],
                    "spatial": {"position": "center", "orientation": "upright", "connects_to": ["walls_upper"]},
                },
                {
                    "name": "roof",
                    "type": "Roof Tiles",
                    "parts": [
                        {"part_id": "3037", "name": "Slope 45 2x4", "category": "Roof Tiles", "color": "Bright Orange", "color_hex": "#FE8A18", "is_trans": False, "quantity": 8},
                        {"part_id": "3038", "name": "Slope 45 2x3", "category": "Roof Tiles", "color": "Bright Orange", "color_hex": "#FE8A18", "is_trans": False, "quantity": 4},
                        {"part_id": "3048", "name": "Slope 45 1x2 Triple", "category": "Roof Tiles", "color": "Dark Red", "color_hex": "#720E0F", "is_trans": False, "quantity": 2},
                    ],
                    "spatial": {"position": "top", "orientation": "angled", "connects_to": ["walls_upper"]},
                },
            ],
            "build_hints": [
                "Start with the green base plate",
                "Build the lower walls with alternating red and white bricks",
                "Add upper walls, leaving gaps for windows",
                "Insert window frames and panes",
                "Attach the orange roof slopes last",
            ],
        }

        steps = json_to_steps(description)
        validation = asdict(StabilityChecker().validate(description))

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "mock-dev-v1",
                "generation_time_ms": 42,
                "json_valid": True,
                "errors": [],
            },
            "validation": validation,
        }


class LegoGenPipeline:
    """Full inference pipeline from image to validated JSON description."""

    def __init__(
        self,
        adapter_path: str | Path | None = None,
        model_name: str = MODEL_NAME,
    ):
        import torch
        from backend.models.vision_encoder import LegoVisionEncoder

        if adapter_path is None:
            adapter_path = CHECKPOINT_DIR / "qwen-lego-lora"

        adapter_path = Path(adapter_path)
        load_adapter = str(adapter_path) if adapter_path.exists() else None

        self.encoder = LegoVisionEncoder(
            model_name=model_name,
            load_adapter=load_adapter,
        )
        self.model = self.encoder.get_model()
        self.processor = self.encoder.get_processor()
        self.model.eval()

    def describe_image(
        self,
        image,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> dict:
        """Generate a structured JSON description from a LEGO image."""
        import torch
        import json
        from backend.models.tokenizer import build_chat_messages, extract_json_from_text
        from backend.inference.constraint_engine import safe_parse_and_validate

        start = time.time()

        with torch.inference_mode():
            # Build chat messages for Qwen3-VL
            messages = build_chat_messages()

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

            # Decode only the generated tokens (skip the input)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        # Parse and validate — handle truncated output from token limit
        parsed = extract_json_from_text(raw_output)
        if parsed:
            reparsed, errors = safe_parse_and_validate(json.dumps(parsed))
            description = reparsed or parsed
            is_valid = reparsed is not None and len(errors) == 0
        else:
            description, errors = safe_parse_and_validate(raw_output)
            is_valid = description is not None
            if is_valid:
                errors = [e for e in errors if "Missing field" not in e]

        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "description": description or {},
            "raw_output": raw_output,
            "is_valid": is_valid,
            "errors": errors,
            "generation_time_ms": elapsed_ms,
        }

    def generate_build(self, image) -> dict:
        """Full pipeline: image -> description -> build steps."""
        result = self.describe_image(image)

        from backend.inference.postprocess_manual import json_to_steps
        from backend.inference.stability_checker import StabilityChecker
        from dataclasses import asdict

        description = result["description"]
        steps = json_to_steps(description) if description else []
        validation = asdict(StabilityChecker().validate(description))

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "qwen3vl-lego-lora-v1",
                "generation_time_ms": result["generation_time_ms"],
                "json_valid": result["is_valid"],
                "errors": result["errors"],
            },
            "validation": validation,
        }


class PlannerPipeline:
    """Full inference pipeline from text prompt to validated JSON description."""

    def __init__(
        self,
        adapter_path: str | Path | None = None,
        model_name: str = PLANNER_MODEL_NAME,
    ):
        import torch
        from backend.models.planner_lm import LegoPlannerLM

        if adapter_path is None:
            adapter_path = PLANNER_CHECKPOINT_DIR

        adapter_path = Path(adapter_path)
        load_adapter = str(adapter_path) if adapter_path.exists() else None

        self.planner = LegoPlannerLM(
            model_name=model_name,
            load_adapter=load_adapter,
        )
        self.model = self.planner.get_model()
        self.tokenizer = self.planner.get_tokenizer()
        self.model.eval()

    def describe_from_text(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> dict:
        """Generate a structured JSON description from a text prompt."""
        import torch
        import json
        from backend.models.tokenizer import build_planner_chat_messages, extract_json_from_text, strip_thinking_blocks
        from backend.inference.constraint_engine import safe_parse_and_validate

        start = time.time()

        with torch.inference_mode():
            messages = build_planner_chat_messages(prompt)

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        # Strip any thinking blocks and parse
        raw_output = strip_thinking_blocks(raw_output)

        # Try exact JSON extraction first, fall back to truncation-aware repair
        parsed = extract_json_from_text(raw_output)
        if parsed:
            reparsed, errors = safe_parse_and_validate(json.dumps(parsed))
            description = reparsed or parsed
            is_valid = reparsed is not None and len(errors) == 0
        else:
            # Output likely truncated by token limit — repair and validate
            description, errors = safe_parse_and_validate(raw_output)
            is_valid = description is not None
            if is_valid:
                # Truncated but repaired — note it but don't treat as failure
                errors = [e for e in errors if "Missing field" not in e]

        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "description": description or {},
            "raw_output": raw_output,
            "is_valid": is_valid,
            "errors": errors,
            "generation_time_ms": elapsed_ms,
        }

    def generate_build(self, prompt: str) -> dict:
        """Full pipeline: text prompt -> description -> build steps."""
        result = self.describe_from_text(prompt)

        from backend.inference.postprocess_manual import json_to_steps
        from backend.inference.stability_checker import StabilityChecker
        from dataclasses import asdict

        description = result["description"]
        steps = json_to_steps(description) if description else []
        validation = asdict(StabilityChecker().validate(description))

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "qwen35-lego-planner-lora-v1",
                "generation_time_ms": result["generation_time_ms"],
                "json_valid": result["is_valid"],
                "errors": result["errors"],
            },
            "validation": validation,
        }

    generate_build_from_text = generate_build


class UnifiedPipeline:
    """Single-model pipeline handling both image->JSON and text->JSON.

    Integrates three caching layers (when CACHE_ENABLED):
      1. KV prefix cache -- reuses pre-computed KV states for static system prompts
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
            # No fallback — vision-only adapter (Qwen3-VL-8B) is incompatible
            # with unified model (Qwen3.5-9B). Run without adapter (base model).
            print(f"Unified checkpoint not found at {adapter_path}, using base model")
            load_adapter = None

        self.wrapper = LegoUnifiedModel(
            model_name=model_name,
            load_adapter=load_adapter,
        )
        self.model = self.wrapper.get_model()
        self.processor = self.wrapper.get_processor()
        self.model.eval()

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
        import torch
        from backend.models.tokenizer import (
            SYSTEM_PROMPT,
            PLANNER_SYSTEM_PROMPT,
            build_chat_messages,
            build_planner_chat_messages,
        )

        start = time.time()

        # Layer 1: KV prefix cache
        if CACHE_KV_PREFIX_ENABLED:
            try:
                self.kv_cache.warmup(self.model, self.processor)
            except Exception as e:
                print(f"[cache] KV prefix warmup failed (falling back to no-cache): {e}")

        # Layer 3: Tokenization cache for static prompt templates
        if CACHE_TOKENIZATION_ENABLED:
            self.tokenization_cache.get_or_compute(
                "vision_template",
                lambda: self.processor.apply_chat_template(
                    build_chat_messages(), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                ),
            )
            self.tokenization_cache.get_or_compute(
                "planner_template_prefix",
                lambda: self.processor.apply_chat_template(
                    [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}],
                    tokenize=False, add_generation_prompt=False,
                ),
            )

        elapsed = int((time.time() - start) * 1000)
        print(f"[cache] Warmup complete in {elapsed}ms")

    def describe_image(
        self,
        image,
        max_new_tokens: int = MAX_NEW_TOKENS,
        cache_key: str | None = None,
    ) -> dict:
        """Generate a structured JSON description from a LEGO image."""
        import torch
        import json
        from backend.models.tokenizer import build_chat_messages, extract_json_from_text, strip_thinking_blocks
        from backend.inference.constraint_engine import safe_parse_and_validate

        # Layer 2: Check response cache
        if CACHE_ENABLED and CACHE_RESPONSE_ENABLED and cache_key:
            cached = self.response_cache.get(cache_key)
            if cached is not None:
                return {**cached, "generation_time_ms": 0, "cached": True}

        start = time.time()

        with torch.inference_mode():
            # Layer 3: Use cached template string or compute fresh
            if CACHE_ENABLED and CACHE_TOKENIZATION_ENABLED:
                text = self.tokenization_cache.get("vision_template")
            else:
                text = None

            if text is None:
                messages = build_chat_messages()
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(self.model.device)

            # Layer 1: KV prefix cache for vision path
            # Note: For vision, image tokens appear after the system prompt in
            # the sequence. We can still use the system-prompt KV prefix since
            # causal attention means earlier positions don't attend to later ones.
            # However, the processor merges image placeholders into input_ids,
            # so we need to strip the prefix tokens and pass only the remainder.
            generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

            if (
                CACHE_ENABLED
                and CACHE_KV_PREFIX_ENABLED
                and self.kv_cache.is_ready
            ):
                kv_clone, prefix_len = self.kv_cache.get_vision_prefix()
                if kv_clone is not None and inputs["input_ids"].shape[1] > prefix_len:
                    # Only pass tokens after the cached prefix
                    new_input_ids = inputs["input_ids"][:, prefix_len:]
                    full_len = inputs["input_ids"].shape[1]
                    attention_mask = torch.ones(
                        (1, full_len), dtype=torch.long, device=self.model.device
                    )
                    position_ids = torch.arange(
                        prefix_len, full_len, dtype=torch.long, device=self.model.device
                    ).unsqueeze(0)

                    # Build kwargs for generate with cached prefix
                    generate_kwargs["past_key_values"] = kv_clone
                    generate_kwargs["attention_mask"] = attention_mask

                    # Pass pixel_values and image_grid_thw if present
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
                    # generated_ids: skip the new_input_ids portion
                    generated_ids = outputs[0][new_input_ids.shape[1]:]
                else:
                    # Prefix longer than input — fall back to standard path
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            else:
                outputs = self.model.generate(**inputs, **generate_kwargs)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

            raw_output = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        raw_output = strip_thinking_blocks(raw_output)

        parsed = extract_json_from_text(raw_output)
        if parsed:
            reparsed, errors = safe_parse_and_validate(json.dumps(parsed))
            description = reparsed or parsed
            is_valid = reparsed is not None and len(errors) == 0
        else:
            description, errors = safe_parse_and_validate(raw_output)
            is_valid = description is not None
            if is_valid:
                errors = [e for e in errors if "Missing field" not in e]

        elapsed_ms = int((time.time() - start) * 1000)

        result = {
            "description": description or {},
            "raw_output": raw_output,
            "is_valid": is_valid,
            "errors": errors,
            "generation_time_ms": elapsed_ms,
            "cached": False,
        }

        # Layer 2: Store in response cache
        if CACHE_ENABLED and CACHE_RESPONSE_ENABLED and cache_key:
            self.response_cache.put(cache_key, result)

        return result

    def describe_from_text(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> dict:
        """Generate a structured JSON description from a text prompt."""
        import torch
        import json
        from backend.models.tokenizer import build_planner_chat_messages, extract_json_from_text, strip_thinking_blocks
        from backend.inference.constraint_engine import safe_parse_and_validate

        # Layer 2: Check response cache
        if CACHE_ENABLED and CACHE_RESPONSE_ENABLED:
            from backend.inference.cache import ResponseCache
            rkey = ResponseCache.make_key_text(prompt)
            cached = self.response_cache.get(rkey)
            if cached is not None:
                return {**cached, "generation_time_ms": 0, "cached": True}
        else:
            rkey = None

        start = time.time()

        with torch.inference_mode():
            messages = build_planner_chat_messages(prompt)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            ).to(self.model.device)

            generate_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

            # Layer 1: KV prefix cache for planner path
            if (
                CACHE_ENABLED
                and CACHE_KV_PREFIX_ENABLED
                and self.kv_cache.is_ready
            ):
                kv_clone, prefix_len = self.kv_cache.get_planner_prefix()
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

                    outputs = self.model.generate(
                        input_ids=new_input_ids,
                        position_ids=position_ids,
                        **generate_kwargs,
                    )
                    generated_ids = outputs[0][new_input_ids.shape[1]:]
                else:
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            else:
                outputs = self.model.generate(**inputs, **generate_kwargs)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

            raw_output = self.processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        raw_output = strip_thinking_blocks(raw_output)

        parsed = extract_json_from_text(raw_output)
        if parsed:
            reparsed, errors = safe_parse_and_validate(json.dumps(parsed))
            description = reparsed or parsed
            is_valid = reparsed is not None and len(errors) == 0
        else:
            description, errors = safe_parse_and_validate(raw_output)
            is_valid = description is not None
            if is_valid:
                errors = [e for e in errors if "Missing field" not in e]

        elapsed_ms = int((time.time() - start) * 1000)

        result = {
            "description": description or {},
            "raw_output": raw_output,
            "is_valid": is_valid,
            "errors": errors,
            "generation_time_ms": elapsed_ms,
            "cached": False,
        }

        # Layer 2: Store in response cache
        if CACHE_ENABLED and CACHE_RESPONSE_ENABLED and rkey:
            self.response_cache.put(rkey, result)

        return result

    def generate_build(self, image, cache_key: str | None = None) -> dict:
        """Full pipeline: image -> description -> build steps."""
        result = self.describe_image(image, cache_key=cache_key)

        from backend.inference.postprocess_manual import json_to_steps
        from backend.inference.stability_checker import StabilityChecker
        from dataclasses import asdict

        description = result["description"]
        steps = json_to_steps(description) if description else []
        validation = asdict(StabilityChecker().validate(description))

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "qwen35-lego-unified-v1",
                "generation_time_ms": result["generation_time_ms"],
                "json_valid": result["is_valid"],
                "errors": result["errors"],
                "cached": result.get("cached", False),
            },
            "validation": validation,
        }

    def generate_build_from_text(self, prompt: str) -> dict:
        """Full pipeline: text prompt -> description -> build steps."""
        result = self.describe_from_text(prompt)

        from backend.inference.postprocess_manual import json_to_steps
        from backend.inference.stability_checker import StabilityChecker
        from dataclasses import asdict

        description = result["description"]
        steps = json_to_steps(description) if description else []
        validation = asdict(StabilityChecker().validate(description))

        return {
            "description": description,
            "steps": steps,
            "metadata": {
                "model_version": "qwen35-lego-unified-v1",
                "generation_time_ms": result["generation_time_ms"],
                "json_valid": result["is_valid"],
                "errors": result["errors"],
                "cached": result.get("cached", False),
            },
            "validation": validation,
        }
