"""Use Gemini (vision) to QC every sourced image against its English label.

Wikipedia/Commons text search occasionally resolves to the wrong sense of a
word (a band called "Belly" instead of a stomach) — the fix for a specific
miss belongs in data/search_hints.json, but checking 270+ images by eye doesn't
scale. This asks Gemini to look at each image and judge whether it actually,
unambiguously depicts the word, and records the verdict in the manifest
instead of silently trusting the search result.

Writes the verdict back into data/images_manifest.json as:
    "gemini_verified": true | false
    "gemini_note": "<short reason, present when false or uncertain>"

Usage:
    export GEMINI_API_KEY=...
    python3 00b_verify_images.py            # only unverified entries
    python3 00b_verify_images.py --refresh  # re-check everything
"""
from __future__ import annotations

import argparse
import asyncio
import json

from common import DATA, IMAGES, client

from google.genai import errors as genai_errors
from google.genai import types

CONCURRENCY = 8
VERDICT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "depicts_concept": types.Schema(type=types.Type.BOOLEAN),
        "reason": types.Schema(type=types.Type.STRING),
    },
    required=["depicts_concept", "reason"],
)
_RETRYABLE = (genai_errors.APIError, asyncio.TimeoutError, ConnectionError)


def prompt_for(english: str, category: str) -> str:
    return (
        f'The attached image is meant to depict the English concept "{english}" '
        f'(BU 200 Word Project category: {category}). Look only at what the image '
        "actually shows. Does it clearly and unambiguously depict that concept, "
        "such that someone could correctly guess the word from the picture alone? "
        "Answer false if the image shows something else entirely (e.g. a person or "
        "place that happens to share the word as a proper name), is a diagram/label "
        "overlay rather than a photo of the thing, or is otherwise misleading."
    )


async def verify_one(aio, english: str, rec: dict, sem: asyncio.Semaphore) -> dict:
    img_path = IMAGES / f"{rec['slug']}.jpg"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/jpeg"),
                types.Part.from_text(text=prompt_for(english, rec["category"])),
            ],
        )
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


async def main_async(refresh: bool, only_low: bool):
    manifest_path = DATA / "images_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    todo = {
        k: v for k, v in manifest.items()
        if (refresh or "gemini_verified" not in v)
        and (not only_low or v.get("confidence") == "low")
    }
    print(f"checking {len(todo)}/{len(manifest)} images")
    if not todo:
        return

    c = client()
    sem = asyncio.Semaphore(CONCURRENCY)
    items = list(todo.items())
    tasks = [asyncio.create_task(verify_one(c.aio, english, rec, sem)) for english, rec in items]
    results = await asyncio.gather(*tasks)

    n_fail = n_ok = n_err = 0
    for (english, _), verdict in zip(items, results):
        manifest[english].update(verdict)
        if verdict["gemini_verified"] is None:
            n_err += 1
        elif verdict["gemini_verified"]:
            n_ok += 1
        else:
            n_fail += 1
            print(f"  FLAGGED  {english:20s} {manifest[english].get('page_title', ''):40s} "
                  f"-- {verdict['gemini_note']}")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nverified_ok={n_ok} flagged={n_fail} errors={n_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-check images already verified")
    ap.add_argument("--only-low", action="store_true", help="only check confidence=low entries")
    args = ap.parse_args()
    asyncio.run(main_async(args.refresh, args.only_low))


if __name__ == "__main__":
    main()
