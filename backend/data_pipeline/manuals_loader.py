"""Downloads LEGO set data and images from the Rebrickable API."""

import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    REBRICKABLE_API_KEY,
    REBRICKABLE_BASE_URL,
    REBRICKABLE_RATE_LIMIT,
    IMAGES_DIR,
    MIN_PARTS,
    MAX_PARTS,
)


class RebrickableLoader:
    def __init__(self, api_key: str = REBRICKABLE_API_KEY):
        self.api_key = api_key
        self.headers = {"Authorization": f"key {self.api_key}"}
        self._last_request_time = 0.0

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

    # ── Sets ───────────────────────────────────────────────────────────

    def fetch_all_sets(
        self,
        min_parts: int = MIN_PARTS,
        max_parts: int = MAX_PARTS,
        max_sets: int | None = None,
    ) -> list[dict]:
        """Paginate through all sets within the part count range."""
        all_sets = []
        page = 1
        while True:
            data = self._get(
                "sets/",
                {
                    "page": page,
                    "page_size": 100,
                    "min_parts": min_parts,
                    "max_parts": max_parts,
                    "ordering": "-year",
                },
            )
            all_sets.extend(data["results"])
            if max_sets and len(all_sets) >= max_sets:
                all_sets = all_sets[:max_sets]
                break
            if not data.get("next"):
                break
            page += 1
        return all_sets

    def fetch_set_inventory(self, set_num: str) -> list[dict]:
        """Get the parts inventory for a specific set."""
        parts = []
        page = 1
        while True:
            data = self._get(
                f"sets/{set_num}/parts/",
                {"page": page, "page_size": 200},
            )
            parts.extend(data["results"])
            if not data.get("next"):
                break
            page += 1
        return parts

    def fetch_themes(self) -> dict[int, dict]:
        """Fetch the full theme hierarchy. Returns {id: {name, parent_id}}."""
        themes = {}
        page = 1
        while True:
            data = self._get("themes/", {"page": page, "page_size": 200})
            for t in data["results"]:
                themes[t["id"]] = {
                    "name": t["name"],
                    "parent_id": t.get("parent_id"),
                }
            if not data.get("next"):
                break
            page += 1
        return themes

    # ── Images ─────────────────────────────────────────────────────────

    def download_set_image(
        self, set_num: str, image_url: str, save_dir: Path = IMAGES_DIR
    ) -> Path | None:
        """Download the set image. Returns the saved file path or None on failure."""
        if not image_url:
            return None
        save_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(image_url).suffix or ".jpg"
        save_path = save_dir / f"{set_num}{ext}"

        if save_path.exists():
            return save_path

        try:
            self._rate_limit()
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            return save_path
        except requests.RequestException:
            return None

    # ── Convenience ────────────────────────────────────────────────────

    def resolve_theme_hierarchy(
        self, theme_id: int, themes: dict[int, dict]
    ) -> list[str]:
        """Walk up the theme tree and return [root, ..., leaf] names."""
        chain = []
        current = theme_id
        while current and current in themes:
            chain.append(themes[current]["name"])
            current = themes[current].get("parent_id")
        chain.reverse()
        return chain
