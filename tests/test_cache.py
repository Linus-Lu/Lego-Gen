"""Tests for three-layer caching system."""

import time


def test_response_cache_put_get():
    """Basic put/get with LRU eviction."""
    from backend.inference.cache import ResponseCache

    cache = ResponseCache(max_size=3, ttl_seconds=60)

    cache.put("a", {"result": 1})
    cache.put("b", {"result": 2})
    cache.put("c", {"result": 3})

    assert cache.get("a") == {"result": 1}
    assert cache.get("b") == {"result": 2}
    assert cache.get("c") == {"result": 3}
    assert cache.get("missing") is None


def test_response_cache_lru_eviction():
    """Oldest entry is evicted when capacity exceeded."""
    from backend.inference.cache import ResponseCache

    cache = ResponseCache(max_size=2, ttl_seconds=60)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # should evict "a"

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_response_cache_ttl_expiry():
    """Entries expire after TTL."""
    from backend.inference.cache import ResponseCache

    cache = ResponseCache(max_size=10, ttl_seconds=0)  # 0s TTL = immediate expiry

    cache.put("key", "value")
    # Entry should be expired immediately
    time.sleep(0.01)
    assert cache.get("key") is None


def test_response_cache_stats():
    """Cache stats track hits and misses."""
    from backend.inference.cache import ResponseCache

    cache = ResponseCache(max_size=10, ttl_seconds=60)

    cache.put("a", 1)
    cache.get("a")      # hit
    cache.get("a")      # hit
    cache.get("miss")   # miss

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["hit_rate"] == 2 / 3


def test_response_cache_clear():
    """Clear resets the cache."""
    from backend.inference.cache import ResponseCache

    cache = ResponseCache(max_size=10, ttl_seconds=60)
    cache.put("a", 1)
    cache.clear()

    assert cache.get("a") is None
    assert cache.stats()["size"] == 0


def test_response_cache_make_key():
    """Hash-based key generation is deterministic."""
    from backend.inference.cache import ResponseCache

    key1 = ResponseCache.make_key_text("hello world")
    key2 = ResponseCache.make_key_text("hello world")
    key3 = ResponseCache.make_key_text("different")

    assert key1 == key2
    assert key1 != key3


def test_tokenization_cache():
    """Tokenization cache stores and retrieves computed values."""
    from backend.inference.cache import TokenizationCache

    cache = TokenizationCache()
    call_count = 0

    def expensive_fn():
        nonlocal call_count
        call_count += 1
        return "computed_result"

    result1 = cache.get_or_compute("key", expensive_fn)
    result2 = cache.get_or_compute("key", expensive_fn)

    assert result1 == "computed_result"
    assert result2 == "computed_result"
    assert call_count == 1  # should only compute once

    assert cache.get("key") == "computed_result"
    assert cache.get("missing") is None


def test_kv_prefix_cache_initial_state():
    """KV prefix cache starts empty."""
    from backend.inference.cache import KVPrefixCache

    cache = KVPrefixCache()

    assert not cache.is_ready
    prefix, length = cache.get_stage1_prefix()
    assert prefix is None
    assert length == 0
