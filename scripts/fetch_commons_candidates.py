"""Fetch up to 50 Wikimedia Commons image candidates per English word, once
per word (shared across all 9 languages, pooled from country-specific searches).

For each word:
1. Identifies the primary countries where the word's BU 200 reference languages
   are spoken (e.g. Akan Twi -> Ghana, Hausa/Igbo -> Nigeria, Swahili -> Kenya/Tanzania,
   Amharic -> Ethiopia, Wolof -> Senegal, Mandinka -> Gambia, isiXhosa/Zulu -> South Africa).
2. Searches Wikimedia Commons for "<word> <descriptor> <country>" across the reference
   countries.
3. Round-robin interleaves the results across countries to produce a balanced pool
   of up to 50 localized candidate images for that concept.
4. Falls back to "<word> Africa" only if country queries produce fewer than 20 hits.

No automated accept/reject step here: a human reviews candidates and picks the
matching image in scripts/commons_image_picker.html.

Usage:
    python3 fetch_commons_candidates.py [--limit N] [--refresh]
Writes data/commons_candidates_africa.json:
    { "<english word>": {
        "countries_searched": "...",
        "category": "...",
        "images": [{"image_url", "page_title", "source_url", "license", "artist", "country", "query"}, ...]
    } }
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from common import DATA, EXCLUDED_CATEGORIES

UA = "vl-mmlu-afri/0.1 (research benchmark; contact: michseth@ghananlp.org)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MIN_SIDE = 200
BAD_TITLE_RE = re.compile(
    r"\b(flag|map of|icon|logo|disambig|location map)\b", re.I
)

LANGUAGE_COUNTRY = {
    "Akan Twi": "Ghana",
    "Hausa": "Nigeria",
    "Igbo": "Nigeria",
    "kiSwahili": "Kenya",
    "Amharic": "Ethiopia",
    "Wolof": "Senegal",
    "Mandinka": "Gambia",
    "isiXhosa": "South Africa",
    "isiZulu": "South Africa",
}

ALL_COUNTRIES = [
    "Ghana", "Nigeria", "Kenya", "South Africa",
    "Ethiopia", "Senegal", "Tanzania", "Gambia",
]

CATEGORY_DESCRIPTOR = {
    "animals": "animal",
    "clothing": "clothing",
    "cooking": "cooking",
    "farm": "farm",
    "food and drink": "food",
    "house": "home",
    "natural world": "nature",
    "professions": "profession",
    "transportation": "vehicle",
}

WORD_DESCRIPTOR = {
    "Apple": "fruit", "Banana": "fruit", "Baobab fruit": "fruit", "Grape": "fruit",
    "Guava": "fruit", "Kolanut": "fruit", "Lemon": "fruit", "Mango": "fruit",
    "Orange": "fruit", "Peach": "fruit", "Pineapple": "fruit", "Plantain": "fruit",
    "Watermelon": "fruit", "Cola nut": "nut",
    "Yam": "vegetable", "Cocoyam": "vegetable", "Onion": "vegetable",
    "Beans": "crop", "Corn": "crop", "Millet": "crop", "Rice": "crop",
    "Water": "drink", "Beer": "drink", "Coffee": "drink",
    "Book": "school supply", "Chalk": "school supply", "Eraser": "school supply",
    "Paper": "school supply", "Pen": "school supply", "Pencil": "school supply",
    "Cutlass": "farm", "Plough": "farm", "Plow": "farm", "Hoe": "farm",
    "Corral": "cattle", "Sour milk": "food", "Cooking oil": "kitchen",
    "Traditional skirt": "clothing", "Western skirt": "clothing",
    "Short-sleeved shirt": "shirt", "Elderly person": "elder",
}


def http_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1)
    return None


def commons_search(query: str, limit: int = 12) -> list[dict]:
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 500,
    }
    data = http_json(COMMONS_API + "?" + urllib.parse.urlencode(params))
    pages = ((data or {}).get("query") or {}).get("pages") or {}
    out = []
    for p in pages.values():
        title = p.get("title", "")
        if BAD_TITLE_RE.search(title):
            continue
        info = (p.get("imageinfo") or [None])[0]
        if not info or info.get("mime") == "image/svg+xml":
            continue
        if info.get("width", 0) < MIN_SIDE or info.get("height", 0) < MIN_SIDE:
            continue
        extmeta = info.get("extmetadata") or {}
        out.append({
            "image_url": info.get("thumburl") or info["url"],
            "page_title": title,
            "source_url": info.get("descriptionurl"),
            "license": (extmeta.get("LicenseShortName") or {}).get("value"),
            "artist": re.sub(r"<[^>]+>", "", (extmeta.get("Artist") or {}).get("value") or ""),
        })
    return out


def fetch_country_candidates_for_word(w: dict) -> tuple[str, dict]:
    english = w["english"]
    primary = []
    for lang in w.get("reference_translations", {}):
        c = LANGUAGE_COUNTRY.get(lang)
        if c and c not in primary:
            primary.append(c)
        if lang == "kiSwahili" and "Tanzania" not in primary:
            primary.append("Tanzania")
    secondary = [c for c in ALL_COUNTRIES if c not in primary]
    country_order = primary + secondary

    descriptor = WORD_DESCRIPTOR.get(english) or CATEGORY_DESCRIPTOR.get(w["category"])
    if descriptor and descriptor.lower() in english.lower():
        descriptor = None

    country_results: dict[str, list[dict]] = {}
    for country in country_order:
        limit = 12 if country in primary else 6
        query = f"{english} {descriptor} {country}" if descriptor else f"{english} {country}"
        hits = commons_search(query, limit=limit)
        if not hits and descriptor:
            query = f"{english} {country}"
            hits = commons_search(query, limit=limit)
        for h in hits:
            h["country"] = country
            h["query"] = query
        country_results[country] = hits
        time.sleep(0.05)

    # Round-robin interleave across countries to keep results diverse
    pooled = []
    seen_urls = set()
    max_len = max((len(h) for h in country_results.values()), default=0)
    for idx in range(max_len):
        for c in country_order:
            if idx < len(country_results[c]):
                item = country_results[c][idx]
                if item["image_url"] not in seen_urls:
                    seen_urls.add(item["image_url"])
                    pooled.append(item)
                    if len(pooled) >= 50:
                        break
        if len(pooled) >= 50:
            break

    # Fallback to broader Africa query if country queries together yielded very few
    if len(pooled) < 20:
        query = f"{english} {descriptor} Africa" if descriptor else f"{english} Africa"
        fallback_hits = commons_search(query, limit=50 - len(pooled))
        if not fallback_hits and descriptor:
            fallback_hits = commons_search(f"{english} Africa", limit=50 - len(pooled))
        for h in fallback_hits:
            if h["image_url"] not in seen_urls:
                seen_urls.add(h["image_url"])
                h["country"] = "Africa (broad)"
                h["query"] = query
                pooled.append(h)
                if len(pooled) >= 50:
                    break

    summary = ", ".join(f"{c} ({len(h)})" for c, h in country_results.items() if h)
    rec = {
        "countries_searched": ", ".join(country_order),
        "primary_countries": primary,
        "category": w["category"],
        "images": pooled,
        "country_summary": summary,
    }
    return english, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    bu200 = json.loads((DATA / "bu200.json").read_text(encoding="utf-8"))
    seen = set()
    words = []
    for w in bu200["words"]:
        if w["category"] in EXCLUDED_CATEGORIES or w["english"] in seen:
            continue
        seen.add(w["english"])
        words.append(w)
    if args.limit:
        words = words[: args.limit]

    out_path = DATA / "commons_candidates_africa.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() and not args.refresh else {}

    # Identify words to fetch
    todo = [
        w for w in words
        if args.refresh
        or w["english"] not in existing
        or not existing[w["english"]].get("images")
        or not existing[w["english"]].get("country_summary")  # re-fetch old Africa-only records
    ]
    print(f"{len(words)} words total: {len(words) - len(todo)} already have country pools, {len(todo)} to fetch")

    if not todo:
        print("All words already fetched with country pools.")
        return

    done_count = len(words) - len(todo)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_country_candidates_for_word, w): w for w in todo}
        for fut in as_completed(futures):
            english, rec = fut.result()
            existing[english] = rec
            done_count += 1
            n_imgs = len(rec["images"])
            countries = rec.get("country_summary", "")
            print(f"[{done_count}/{len(words)}] {english:18s} : {n_imgs:2d} images | {countries}")
            # Write periodically to keep file updated
            out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone! Wrote {len(existing)} words -> {out_path}")


if __name__ == "__main__":
    main()
