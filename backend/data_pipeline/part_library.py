"""Manages the LEGO part and color catalog from Rebrickable, with local caching."""

import json
import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    REBRICKABLE_API_KEY,
    REBRICKABLE_BASE_URL,
    REBRICKABLE_RATE_LIMIT,
    CACHE_DIR,
)


class PartLibrary:
    def __init__(self, api_key: str = REBRICKABLE_API_KEY):
        self.api_key = api_key
        self.headers = {"Authorization": f"key {self.api_key}"}
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.categories: dict[int, str] = {}  # id -> name
        self.colors: dict[int, dict] = {}  # id -> {name, rgb, is_trans}
        self._last_request_time = 0.0

    # ── API helpers ────────────────────────────────────────────────────

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < REBRICKABLE_RATE_LIMIT:
            time.sleep(REBRICKABLE_RATE_LIMIT - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        self._rate_limit()
        url = f"{REBRICKABLE_BASE_URL}/{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Fetch & cache ──────────────────────────────────────────────────

    def fetch_categories(self) -> dict[int, str]:
        """Fetch all part categories from Rebrickable and cache locally."""
        results = []
        url_params = {"page_size": 1000}
        data = self._get("part_categories/", url_params)
        results.extend(data["results"])

        self.categories = {c["id"]: c["name"] for c in results}
        self._save_cache("categories.json", self.categories)
        return self.categories

    def fetch_colors(self) -> dict[int, dict]:
        """Fetch all colors from Rebrickable and cache locally."""
        results = []
        page = 1
        while True:
            data = self._get("colors/", {"page": page, "page_size": 200})
            results.extend(data["results"])
            if not data.get("next"):
                break
            page += 1

        self.colors = {
            c["id"]: {
                "name": c["name"],
                "rgb": c["rgb"],
                "is_trans": c["is_trans"],
            }
            for c in results
        }
        self._save_cache("colors.json", self.colors)
        return self.colors

    def load_cache(self) -> bool:
        """Load categories and colors from local cache. Returns True if successful."""
        cat_path = self.cache_dir / "categories.json"
        col_path = self.cache_dir / "colors.json"
        if cat_path.exists() and col_path.exists():
            with open(cat_path, "r") as f:
                raw = json.load(f)
                self.categories = {int(k): v for k, v in raw.items()}
            with open(col_path, "r") as f:
                raw = json.load(f)
                self.colors = {int(k): v for k, v in raw.items()}
            return True
        return False

    def ensure_loaded(self):
        """Load from cache or fetch from API."""
        if not self.load_cache():
            self.fetch_categories()
            self.fetch_colors()

    def get_color_name(self, color_id: int) -> str:
        return self.colors.get(color_id, {}).get("name", "Unknown")

    def get_color_hex(self, color_id: int) -> str:
        return self.colors.get(color_id, {}).get("rgb", "000000")

    def get_category_name(self, category_id: int) -> str:
        return self.categories.get(category_id, "Unknown")

    # ── Internal ───────────────────────────────────────────────────────

    def _save_cache(self, filename: str, data: dict):
        path = self.cache_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
