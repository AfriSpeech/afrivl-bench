"""Get one representative, verified, Africa-contextual image per BU 200 concept.

Keyword search alone is not trustworthy: a query like "shoulder nigeria" or
"star ghana" still matches brand names, place names and unrelated photos that
merely mention the country. So this never accepts a search hit on sight -
every candidate is checked with Gemini (vision) before being accepted, and a
candidate that fails the check is discarded in favour of the next one.

For each concept (except colors/numbers, which are generated, and "days of the
week", which is skipped - see common.EXCLUDED_CATEGORIES):

  1. Build a candidate list from Wikimedia Commons full-text search, trying
     (in order) any data/search_hints.json override, then "<word> <country>"
     for each of the 8 countries the 9 BU 200 reference languages are actually
     spoken in (COUNTRIES below), then the bare word as a last resort.
  2. Ask Gemini whether each candidate, in order, actually depicts the word.
     The first one that passes is kept.
  3. If nothing passes, the word is left unfetched and printed as NEEDS-HINT
     so a human can add a `data/search_hints.json` entry (a sharper query, or
     a direct "File:..." pin).

Writes data/images/, data/images_manifest.json (each entry records
`gemini_verified` and `gemini_note`, and the query/candidate list it came
from). Re-run is idempotent; pass --refresh to redo words already verified.

Usage:
    export GEMINI_API_KEY=...
    python3 00_fetch_images.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from common import DATA, EXCLUDED_CATEGORIES, IMAGES, client

from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

MANIFEST = DATA / "images_manifest.json"
HINTS_PATH = DATA / "search_hints.json"

UA = "vl-mmlu-afri/0.1 (research benchmark; contact: michseth@ghananlp.org)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

MIN_SIDE = 250
BAD_TITLE_RE = re.compile(
    r"\b(flag|map of|icon|logo|diagram|chart|coat of arms|disambig|location map)\b", re.I
)

# The 9 BU 200 reference languages' primary countries (from afriso): isiXhosa/
# isiZulu -> South Africa, Amharic -> Ethiopia, Wolof -> Senegal, Hausa/Igbo ->
# Nigeria, kiSwahili -> Kenya/Tanzania, Akan Twi -> Ghana, Mandinka -> Gambia.
COUNTRIES = ["ghana", "nigeria", "south africa", "kenya", "ethiopia",
             "senegal", "tanzania", "gambia"]
MAX_CANDIDATES_PER_QUERY = 4
MAX_CANDIDATES_TOTAL = 12

VERDICT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "depicts_concept": types.Schema(type=types.Type.BOOLEAN),
        "reason": types.Schema(type=types.Type.STRING),
    },
    required=["depicts_concept", "reason"],
)
_RETRYABLE = (genai_errors.APIError, asyncio.TimeoutError, ConnectionError)


def slugify(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")


def http_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


# ------------------------------------------------------------- commons search
def commons_search(query: str, limit: int) -> list[dict]:
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 800,
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
            "query_used": query, "page_title": title,
            "source_url": info.get("descriptionurl"),
            "image_url": info.get("thumburl") or info["url"],
            "license": (extmeta.get("LicenseShortName") or {}).get("value"),
            "artist": re.sub(r"<[^>]+>", "", (extmeta.get("Artist") or {}).get("value") or ""),
        })
    return out


def commons_file_direct(file_title: str) -> dict | None:
    params = {
        "action": "query", "format": "json", "titles": file_title,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 800,
    }
    data = http_json(COMMONS_API + "?" + urllib.parse.urlencode(params))
    pages = ((data or {}).get("query") or {}).get("pages") or {}
    for p in pages.values():
        info = (p.get("imageinfo") or [None])[0]
        if not info:
            continue
        extmeta = info.get("extmetadata") or {}
        return {
            "query_used": f"manual:{file_title}", "page_title": p.get("title"),
            "source_url": info.get("descriptionurl"),
            "image_url": info.get("thumburl") or info["url"],
            "license": (extmeta.get("LicenseShortName") or {}).get("value"),
            "artist": re.sub(r"<[^>]+>", "", (extmeta.get("Artist") or {}).get("value") or ""),
        }
    return None


# Human anatomy doesn't vary by country - a nose is a nose everywhere - and
# forcing a country name into the query actively hurt these: "Chin south
# africa" matched a war memorial statue, "Nose south africa" matched an
# airliner's nose cone. Search plain English first for this category, only
# falling back to the country cycle if that finds nothing.
UNIVERSAL_CATEGORIES = {"body parts"}


def build_candidates(english: str, category: str, hints: dict) -> list[dict]:
    hint = hints.get(english)
    candidates: list[dict] = []
    seen_urls: set[str] = set()

    def add_all(cands: list[dict]):
        for c in cands:
            if c["image_url"] not in seen_urls:
                seen_urls.add(c["image_url"])
                candidates.append(c)

    if isinstance(hint, str) and hint.startswith("File:"):
        rec = commons_file_direct(hint)
        if rec:
            add_all([rec])
        return candidates  # a manual pin is trusted outright, nothing else tried
    if hint:
        add_all(commons_search(hint, MAX_CANDIDATES_PER_QUERY))

    if category in UNIVERSAL_CATEGORIES:
        add_all(commons_search(english, MAX_CANDIDATES_PER_QUERY * 2))

    for country in COUNTRIES:
        if len(candidates) >= MAX_CANDIDATES_TOTAL:
            break
        add_all(commons_search(f"{english} {country}", MAX_CANDIDATES_PER_QUERY))
    if len(candidates) < MAX_CANDIDATES_TOTAL:
        add_all(commons_search(english, MAX_CANDIDATES_PER_QUERY))
    return candidates[:MAX_CANDIDATES_TOTAL]


# ------------------------------------------------------------- generated art
def make_color_swatch(word: str) -> tuple[bytes, dict]:
    rgb = COLOR_RGB[word.lower()]
    img = Image.new("RGB", (600, 600), rgb)
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue(), {"source": "generated-color", "rgb": list(rgb)}


def make_number_image(word: str) -> tuple[bytes, dict]:
    digits = NUMBER_WORDS.get(word.lower(), "?")
    img = Image.new("RGB", (600, 600), (250, 250, 250))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=300)
    bbox = d.textbbox((0, 0), digits, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((600 - w) / 2 - bbox[0], (600 - h) / 2 - bbox[1]), digits, fill=(10, 10, 10), font=font)
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue(), {"source": "generated-number", "digits": digits}


# -------------------------------------------------------------- verification
def verify_prompt(english: str, category: str) -> str:
    return (
        f'This image is a candidate for a picture-dictionary entry for the English word '
        f'"{english}" (BU 200 Word Project category: {category}), aimed at someone learning '
        "the word. Picture a person glancing at this image for two seconds and being asked "
        '"what is this a picture of?" - would nearly everyone immediately say '
        f'"{english}", with no further context? Answer false unless ALL of these hold:\n'
        f"1. The {english} is the single, large, unmistakably dominant subject of the "
        "photo - not a small or incidental detail elsewhere in a busy scene (e.g. a tie "
        "glimpsed in a photo of two people shaking hands does NOT count).\n"
        "2. It is the common, concrete, literal sense of the word, not a metaphorical, "
        "idiomatic, brand-name, or proper-noun use (e.g. an airplane's 'nose' is not a "
        "'nose'; 'Africa Star' is not a 'star'), and for a body part specifically, it must "
        "be a HUMAN body part, not an animal's.\n"
        "3. For a whole food/plant/animal word, it shows the thing in its ordinary, "
        "recognizable everyday form - not a prepared dish, a cut/processed product, a "
        "closeup so tight the whole form isn't visible, or an unrelated plant part (e.g. "
        "a plated chicken dish doesn't read as 'chicken'; leaves don't read as the fruit).\n"
        "4. It is an actual photograph, not a diagram, map, flag, logo, or illustration "
        "with text labels overlaid.\n"
        "Be strict and literal-minded; when in doubt, answer false."
    )


async def verify(aio, image_bytes: bytes, english: str, category: str, sem: asyncio.Semaphore) -> dict:
    contents = [
        types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=verify_prompt(english, category)),
        ])
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=VERDICT_SCHEMA, temperature=0,
    )
    async with sem:
        for attempt in range(4):
            try:
                r = await aio.models.generate_content(
                    model="gemini-2.5-flash", contents=contents, config=config
                )
                break
            except _RETRYABLE as e:
                if attempt == 3:
                    return {"gemini_verified": None, "gemini_note": f"error: {e}"}
                await asyncio.sleep(2 * (attempt + 1))
    try:
        v = json.loads(r.text)
        return {"gemini_verified": bool(v["depicts_concept"]), "gemini_note": v["reason"]}
    except Exception:
        return {"gemini_verified": None, "gemini_note": "unparseable response"}


async def resolve_word(aio, w: dict, hints: dict, sem: asyncio.Semaphore) -> tuple[str, dict | None]:
    english = w["english"]
    candidates = build_candidates(english, w["category"], hints)
    if not candidates:
        return english, None
    for i, cand in enumerate(candidates):
        try:
            img_bytes = download(cand["image_url"])
        except Exception as e:
            print(f"  (download failed for {english} candidate {i}: {e})", file=sys.stderr)
            continue
        verdict = await verify(aio, img_bytes, english, w["category"], sem)
        if verdict["gemini_verified"]:
            rec = {**cand, **verdict, "category": w["category"], "english": english,
                   "slug": slugify(english), "_bytes": img_bytes,
                   "candidates_tried": i + 1}
            return english, rec
        print(f"  ...   {english:20s} rejected candidate {i+1}/{len(candidates)}: "
              f"{cand['page_title']} -- {verdict['gemini_note']}")
    return english, None


async def main_async(args):
    bu200 = json.loads((DATA / "bu200.json").read_text(encoding="utf-8"))
    words = [w for w in bu200["words"] if w["category"] not in EXCLUDED_CATEGORIES]
    if args.limit:
        words = words[: args.limit]
    hints = json.loads(HINTS_PATH.read_text(encoding="utf-8")) if HINTS_PATH.exists() else {}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}

    todo = [
        w for w in words
        if args.refresh or w["english"] not in manifest or not manifest[w["english"]].get("gemini_verified")
    ]
    print(f"{len(words) - len(todo)} already verified, resolving {len(todo)} word(s)")

    c = client()
    sem = asyncio.Semaphore(args.concurrency)
    needs_hint = []
    done = 0
    for coro in asyncio.as_completed([resolve_word(c.aio, w, hints, sem) for w in todo]):
        english, rec = await coro
        done += 1
        if rec is None:
            needs_hint.append(english)
            print(f"[{done}/{len(todo)}] NEEDS-HINT  {english}")
            continue
        img_bytes = rec.pop("_bytes")
        (IMAGES / f"{rec['slug']}.jpg").write_bytes(img_bytes)
        manifest[english] = rec
        print(f"[{done}/{len(todo)}] OK  {english:20s} <- {rec['page_title']} "
              f"(tried {rec['candidates_tried']})")

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nverified={len(words) - len(needs_hint)} needs_hint={len(needs_hint)} "
          f"total_in_manifest={len(manifest)}")
    if needs_hint:
        print("\nwords with NO verified image (add a data/search_hints.json entry):")
        for w in needs_hint:
            print(f"    {w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
