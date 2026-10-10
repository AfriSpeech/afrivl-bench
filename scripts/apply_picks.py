"""Apply human picks from commons_image_picker.html (picks.json).

Reads a picks.json file (exported from the picker tool), downloads the chosen
images into data/images/<slug>.jpg, and updates data/images_manifest.json.

Usage:
    python3 apply_picks.py [path/to/picks.json]
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from common import DATA, EXCLUDED_CATEGORIES, IMAGES

UA = "vl-mmlu-afri/0.1 (research benchmark; contact: michseth@ghananlp.org)"


def slugify(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")


def download_image(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                dest.write_bytes(r.read())
                return True
        except Exception as e:
            if attempt == 2:
                print(f"  FAILED to download {url}: {e}", file=sys.stderr)
                return False
    return False


def main():
    picks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "picks.json"
    if not picks_path.exists():
        # check repo root as fallback
        if (DATA.parent / "picks.json").exists():
            picks_path = DATA.parent / "picks.json"
        else:
            print(f"Error: {picks_path} does not exist. Pass path to exported picks.json.", file=sys.stderr)
            sys.exit(1)

    picks = json.loads(picks_path.read_text(encoding="utf-8"))
    manifest_path = DATA / "images_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    bu200 = json.loads((DATA / "bu200.json").read_text(encoding="utf-8"))
    word_cat = {w["english"]: w["category"] for w in bu200["words"]}

    applied = 0
    none_matched = 0

    for english, pick in picks.items():
        if pick.get("none_match") or not pick.get("chosen_url"):
            none_matched += 1
            if english in manifest:
                manifest[english]["gemini_verified"] = False
                manifest[english]["human_verified"] = False
                manifest[english]["review_note"] = "human marked none match"
            continue

        slug = slugify(english)
        dest = IMAGES / f"{slug}.jpg"
        print(f"Downloading [{english}] ({pick.get('country', 'unknown')}) -> {dest.name}...")
        ok = download_image(pick["chosen_url"], dest)
        if not ok:
            continue

        manifest[english] = {
            "english": english,
            "slug": slug,
            "category": word_cat.get(english, "unknown"),
            "image_url": pick["chosen_url"],
            "source_url": pick.get("source_url"),
            "license": pick.get("license"),
            "artist": pick.get("artist"),
            "country": pick.get("country"),
            "human_verified": True,
            "gemini_verified": True,
            "chosen_at": pick.get("ts"),
        }
        applied += 1

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nApplied: {applied} images saved and manifest updated.")
    print(f"None match: {none_matched} words marked.")
    print(f"Total entries in manifest: {len(manifest)}")


if __name__ == "__main__":
    main()
