"""Three-layer prompt caching system for inference acceleration.

Layer 1 - KVPrefixCache:  Pre-computed KV states for static system prompts.
Layer 2 - ResponseCache:  LRU + TTL cache for complete inference results.
Layer 3 - TokenizationCache: Cached chat-template strings for static prompts.
"""

import hashlib
import threading
import time
from collections import OrderedDict


# ── Layer 1: KV-Cache Prefix Caching ─────────────────────────────────


class KVPrefixCache:
    """Caches pre-computed KV states for static system prompt prefixes.

    After warmup, each inference request clones the cached prefix and passes
    it as past_key_values to model.generate(), skipping redundant computation
    of the system-prompt tokens.
    """

    def __init__(self):
        self._stage1_cache = None
        self._stage1_prefix_len: int = 0
        self._warmed_up = False

    def warmup_stage1(self, model, processor, system_prompt: str):
        """Run forward pass on Stage 1 system prompt to populate KV cache."""
        import torch

        device = next(model.parameters()).device
        messages = [{"role": "system", "content": system_prompt}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = processor(text=[text], return_tensors="pt").to(device)

        with torch.inference_mode():
            out = model(**inputs, use_cache=True)
        self._stage1_cache = out.past_key_values
        self._stage1_prefix_len = inputs["input_ids"].shape[1]
        self._warmed_up = True
        print(f"[cache] Stage 1 KV prefix warmup: {self._stage1_prefix_len} tokens")

    def get_stage1_prefix(self):
        """Return (cloned DynamicCache, prefix_length) for Stage 1 path."""
        if self._stage1_cache is None:
            return None, 0
        return self._clone_kv_cache(self._stage1_cache), self._stage1_prefix_len

    @property
    def is_ready(self) -> bool:
        return self._warmed_up

    @staticmethod
    def _clone_kv_cache(cache):
        """Clone a DynamicCache using tensor .clone() instead of copy.deepcopy.

        copy.deepcopy on GPU tensors is very slow because it copies Python
        metadata + pickles/unpickles. Tensor .clone() stays on GPU and is
        ~10-50x faster for the ~48-layer KV cache of a 9B model.
        """
        from transformers.cache_utils import DynamicCache

        cloned = DynamicCache()
        for layer_idx in range(len(cache)):
            cloned.update(
                cache.key_cache[layer_idx].clone(),
                cache.value_cache[layer_idx].clone(),
                layer_idx,
            )
        return cloned


# ── Layer 2: Response Caching ─────────────────────────────────────────


class _CacheEntry:
    __slots__ = ("value", "created_at")

    def __init__(self, value):
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

    def get(self, key: str):
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if (time.monotonic() - entry.created_at) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = _CacheEntry(value)
            else:
                self._cache[key] = _CacheEntry(value)
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
