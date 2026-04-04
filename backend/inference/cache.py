"""Three-layer prompt caching system for inference acceleration.

Layer 1 - KVPrefixCache:  Pre-computed KV states for static system prompts.
Layer 2 - ResponseCache:  LRU + TTL cache for complete inference results.
Layer 3 - TokenizationCache: Cached chat-template strings for static prompts.
"""

import copy
import hashlib
import threading
import time
from collections import OrderedDict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Layer 1: KV-Cache Prefix Caching ─────────────────────────────────


class KVPrefixCache:
    """Caches pre-computed KV states for static system prompt prefixes.

    After warmup, each inference request clones the cached prefix and passes
    it as past_key_values to model.generate(), skipping redundant computation
    of the ~100 system-prompt tokens.
    """

    def __init__(self):
        self._vision_cache = None      # DynamicCache for SYSTEM_PROMPT
        self._planner_cache = None     # DynamicCache for PLANNER_SYSTEM_PROMPT
        self._vision_prefix_len: int = 0
        self._planner_prefix_len: int = 0
        self._warmed_up = False

    def warmup(self, model, processor):
        """Run forward pass on system prompt tokens to populate KV caches.

        Must be called once after model loading, before any inference.
        """
        import torch
        from backend.models.tokenizer import (
            SYSTEM_PROMPT,
            PLANNER_SYSTEM_PROMPT,
        )

        device = next(model.parameters()).device

        # --- Vision prefix (SYSTEM_PROMPT) ---
        vision_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        vision_text = processor.apply_chat_template(
            vision_messages, tokenize=False, add_generation_prompt=False,
        )
        vision_inputs = processor(
            text=[vision_text], return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            vision_out = model(
                **vision_inputs,
                use_cache=True,
            )
        self._vision_cache = vision_out.past_key_values
        self._vision_prefix_len = vision_inputs["input_ids"].shape[1]

        # --- Planner prefix (PLANNER_SYSTEM_PROMPT) ---
        planner_messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        planner_text = processor.apply_chat_template(
            planner_messages, tokenize=False, add_generation_prompt=False,
        )
        planner_inputs = processor(
            text=[planner_text], return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            planner_out = model(
                **planner_inputs,
                use_cache=True,
            )
        self._planner_cache = planner_out.past_key_values
        self._planner_prefix_len = planner_inputs["input_ids"].shape[1]

        self._warmed_up = True
        print(
            f"[cache] KV prefix warmup complete: "
            f"vision={self._vision_prefix_len} tokens, "
            f"planner={self._planner_prefix_len} tokens"
        )

    @property
    def is_ready(self) -> bool:
        return self._warmed_up

    def get_vision_prefix(self):
        """Return (cloned DynamicCache, prefix_length) for vision path."""
        if not self._warmed_up:
            return None, 0
        return copy.deepcopy(self._vision_cache), self._vision_prefix_len

    def get_planner_prefix(self):
        """Return (cloned DynamicCache, prefix_length) for planner path."""
        if not self._warmed_up:
            return None, 0
        return copy.deepcopy(self._planner_cache), self._planner_prefix_len


# ── Layer 2: Response Caching ─────────────────────────────────────────


class _CacheEntry:
    __slots__ = ("value", "created_at")

    def __init__(self, value: dict):
        self.value = value
        self.created_at = time.monotonic()


class ResponseCache:
    """LRU cache with TTL for complete inference results.

    Thread-safe via threading.Lock (FastAPI runs sync handlers in a thread pool).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key_text(prompt: str) -> str:
        return hashlib.sha256(prompt.strip().encode()).hexdigest()

    @staticmethod
    def make_key_image(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            # Check TTL
            if (time.monotonic() - entry.created_at) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: dict):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = _CacheEntry(value)
            else:
                self._cache[key] = _CacheEntry(value)
                # Evict LRU if over capacity
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(self._hits + self._misses, 1),
            }

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# ── Layer 3: Tokenization Caching ────────────────────────────────────


class TokenizationCache:
    """Caches chat-template strings for static system prompts.

    Written once at warmup, read-only thereafter. No lock needed.
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    def get_or_compute(self, key: str, compute_fn) -> str:
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def get(self, key: str) -> str | None:
        return self._cache.get(key)
