"""Generate language-conditioned human evaluation datasets.

Reads bu200.json and commons_candidates_africa.json, and outputs per-language
data files containing items filtered and prioritized for each language's country:
    data/human_eval/<iso>.json
    human-eval/data/<iso>.json

Languages:
    Akan Twi (twi) -> Ghana
    Hausa (hau) -> Nigeria
    Igbo (ibo) -> Nigeria
    kiSwahili (swh) -> Kenya, Tanzania
    Amharic (amh) -> Ethiopia
    Wolof (wol) -> Senegal
    Mandinka (mnk) -> Gambia
    isiXhosa (xho) -> South Africa
    isiZulu (zul) -> South Africa
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import DATA, EXCLUDED_CATEGORIES, ROOT

HUMAN_EVAL_DIR = ROOT / "human-eval" / "data"
LOCAL_DATA_DIR = DATA / "human_eval"

HUMAN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_CONFIG = {
    "Akan Twi": {
        "iso": "twi",
        "flag": "🇬🇭",
        "primary_countries": ["Ghana"],
        "fallback_countries": ["Nigeria", "Senegal", "Gambia", "Kenya", "South Africa"],
    },
    "Hausa": {
        "iso": "hau",
        "flag": "🇳🇬",
        "primary_countries": ["Nigeria"],
        "fallback_countries": ["Ghana", "Senegal", "Kenya", "Gambia", "South Africa"],
    },
    "Igbo": {
        "iso": "ibo",
        "flag": "🇳🇬",
        "primary_countries": ["Nigeria"],
        "fallback_countries": ["Ghana", "Kenya", "Senegal", "South Africa", "Gambia"],
    },
    "kiSwahili": {
        "iso": "swh",
        "flag": "🇰🇪 🇹🇿",
        "primary_countries": ["Kenya", "Tanzania"],
        "fallback_countries": ["Ethiopia", "South Africa", "Uganda", "Ghana", "Nigeria"],
    },
    "Amharic": {
        "iso": "amh",
        "flag": "🇪🇹",
        "primary_countries": ["Ethiopia"],
        "fallback_countries": ["Kenya", "Tanzania", "South Africa", "Ghana", "Nigeria"],
    },
    "Wolof": {
        "iso": "wol",
        "flag": "🇸🇳",
        "primary_countries": ["Senegal"],
        "fallback_countries": ["Gambia", "Ghana", "Nigeria", "Kenya", "South Africa"],
    },
    "Mandinka": {
        "iso": "mnk",
        "flag": "🇬🇲",
        "primary_countries": ["Gambia"],
        "fallback_countries": ["Senegal", "Ghana", "Nigeria", "Kenya", "South Africa"],
    },
    "isiXhosa": {
        "iso": "xho",
        "flag": "🇿🇦",
        "primary_countries": ["South Africa"],
        "fallback_countries": ["Kenya", "Tanzania", "Ghana", "Nigeria", "Ethiopia"],
    },
    "isiZulu": {
        "iso": "zul",
        "flag": "🇿🇦",
        "primary_countries": ["South Africa"],
        "fallback_countries": ["Kenya", "Tanzania", "Ghana", "Nigeria", "Ethiopia"],
    },
}


def slugify(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")


def main():
    commons_data = json.loads((DATA / "commons_candidates_africa.json").read_text(encoding="utf-8"))
    bu200 = json.loads((DATA / "bu200.json").read_text(encoding="utf-8"))

    manifest_languages = []

    for lang, cfg in LANGUAGE_CONFIG.items():
        iso = cfg["iso"]
        items = []
        primary_set = set(cfg["primary_countries"])
        fallback_set = set(cfg["fallback_countries"])

        for w in bu200["words"]:
            if w["category"] in EXCLUDED_CATEGORIES:
                continue
            eng = w["english"]
            if eng not in commons_data:
                continue
            local_word = (w.get("reference_translations") or {}).get(lang)
            if not local_word:
                continue

            all_imgs = commons_data[eng].get("images", [])
            primary_imgs = [im for im in all_imgs if im.get("country") in primary_set]
            fallback_imgs = [im for im in all_imgs if im.get("country") in fallback_set]
            other_imgs = [im for im in all_imgs if im.get("country") not in primary_set and im.get("country") not in fallback_set]

            # Pool: primary country images first, then fallback countries, then others if needed
            pool = primary_imgs[:]
            if len(pool) < 12:
                pool.extend(fallback_imgs[:(12 - len(pool))])
            if len(pool) < 12:
                pool.extend(other_imgs[:(12 - len(pool))])

            slug = slugify(eng)
            items.append({
                "id": f"{iso}-{slug}",
                "english": eng,
                "local_word": local_word,
                "category": w["category"],
                "slug": slug,
                "primary_country": cfg["primary_countries"][0],
                "images": pool[:16],
            })

        # Sort items alphabetically by English concept
        items.sort(key=lambda x: x["english"])

        # Write to both locations
        json_bytes = json.dumps(items, indent=2, ensure_ascii=False)
        (LOCAL_DATA_DIR / f"{iso}.json").write_text(json_bytes, encoding="utf-8")
        (HUMAN_EVAL_DIR / f"{iso}.json").write_text(json_bytes, encoding="utf-8")

        manifest_languages.append({
            "language": lang,
            "iso": iso,
            "flag": cfg["flag"],
            "primary_country": cfg["primary_countries"][0],
            "all_primary_countries": cfg["primary_countries"],
            "word_count": len(items),
        })
        print(f"[{iso}] {lang:10s} ({cfg['flag']} {', '.join(cfg['primary_countries'])}) -> {len(items)} concepts")

    # Write languages manifest
    lang_manifest_bytes = json.dumps(manifest_languages, indent=2, ensure_ascii=False)
    (LOCAL_DATA_DIR / "languages.json").write_text(lang_manifest_bytes, encoding="utf-8")
    (HUMAN_EVAL_DIR / "languages.json").write_text(lang_manifest_bytes, encoding="utf-8")
    print(f"\nSaved languages.json and all {len(manifest_languages)} language datasets.")


if __name__ == "__main__":
    main()
