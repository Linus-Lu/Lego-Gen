"""Tests for the three-layer prompt caching system."""

import threading
import time

import pytest

from backend.inference.cache import KVPrefixCache, ResponseCache, TokenizationCache


# ── ResponseCache Tests ──────────────────────────────────────────────


class TestResponseCache:
    def test_put_and_get(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("key1", {"result": "value1"})
        assert cache.get("key1") == {"result": "value1"}

    def test_miss_returns_none(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = ResponseCache(max_size=10, ttl_seconds=1)
        cache.put("key1", {"result": "value1"})
        assert cache.get("key1") is not None
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        cache = ResponseCache(max_size=3, ttl_seconds=60)
        cache.put("a", {"v": 1})
        cache.put("b", {"v": 2})
        cache.put("c", {"v": 3})
        # Access "a" to make it recently used
        cache.get("a")
        # Adding "d" should evict "b" (least recently used)
        cache.put("d", {"v": 4})
        assert cache.get("b") is None
        assert cache.get("a") == {"v": 1}
        assert cache.get("c") == {"v": 3}
        assert cache.get("d") == {"v": 4}

    def test_overwrite_existing_key(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("key1", {"v": 1})
        cache.put("key1", {"v": 2})
        assert cache.get("key1") == {"v": 2}

    def test_stats(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("key1", {"v": 1})
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        cache = ResponseCache(max_size=10, ttl_seconds=60)
        cache.put("a", {"v": 1})
        cache.put("b", {"v": 2})
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.stats()["size"] == 0

    def test_make_key_text(self):
        key1 = ResponseCache.make_key_text("Build me a red car")
        key2 = ResponseCache.make_key_text("Build me a red car")
        key3 = ResponseCache.make_key_text("Build me a blue car")
        assert key1 == key2
        assert key1 != key3

    def test_make_key_text_strips_whitespace(self):
        key1 = ResponseCache.make_key_text("  hello  ")
        key2 = ResponseCache.make_key_text("hello")
        assert key1 == key2

    def test_make_key_image(self):
        key1 = ResponseCache.make_key_image(b"\x89PNG\r\nfakedata")
        key2 = ResponseCache.make_key_image(b"\x89PNG\r\nfakedata")
        key3 = ResponseCache.make_key_image(b"\x89PNG\r\ndifferent")
        assert key1 == key2
        assert key1 != key3

    def test_thread_safety(self):
        cache = ResponseCache(max_size=100, ttl_seconds=60)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    cache.put(f"key-{start + i}", {"v": start + i})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(200):
                    cache.get(f"key-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(100,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── TokenizationCache Tests ─────────────────────────────────────────


class TestTokenizationCache:
    def test_get_or_compute(self):
        cache = TokenizationCache()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        result1 = cache.get_or_compute("key1", compute)
        result2 = cache.get_or_compute("key1", compute)
        assert result1 == "computed_value"
        assert result2 == "computed_value"
        assert call_count == 1  # Only computed once

    def test_get_returns_none_for_missing(self):
        cache = TokenizationCache()
        assert cache.get("nonexistent") is None

    def test_get_returns_cached_value(self):
        cache = TokenizationCache()
        cache.get_or_compute("key1", lambda: "value1")
        assert cache.get("key1") == "value1"

    def test_different_keys(self):
        cache = TokenizationCache()
        cache.get_or_compute("a", lambda: "val_a")
        cache.get_or_compute("b", lambda: "val_b")
        assert cache.get("a") == "val_a"
        assert cache.get("b") == "val_b"


# ── KVPrefixCache Tests (unit-level, no model) ──────────────────────


class TestKVPrefixCache:
    def test_not_ready_before_warmup(self):
        cache = KVPrefixCache()
        assert not cache.is_ready

    def test_stage1_returns_none_before_warmup(self):
        cache = KVPrefixCache()
        kv, length = cache.get_stage1_prefix()
        assert kv is None
        assert length == 0
