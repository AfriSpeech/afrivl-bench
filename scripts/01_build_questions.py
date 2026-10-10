"""Build the multiple-choice question set.

For each of the 9 BU 200 reference languages, the pool of eligible concepts is
every word that has BOTH an image (data/images_manifest.json) AND a human
reference translation into that language. For each concept in the pool, one
question is built:

    image of <concept>
    "Which word means this in <language>?"
    choices = [correct translation] + (N_CHOICES - 1) translations of OTHER
              concepts in the same pool/language, shuffled.

Distractors are real human translations of other BU 200 words into the same
language, never invented text, so a wrong choice is still a plausible-looking
word in that language. Writes data/questions.json.
"""
from __future__ import annotations

import json
import random

from common import DATA, EXCLUDED_CATEGORIES, LANGUAGE_ISO, N_CHOICES, REFERENCE_LANGUAGES, SEED


def main():
    bu200 = json.loads((DATA / "bu200.json").read_text(encoding="utf-8"))
    manifest = json.loads((DATA / "images_manifest.json").read_text(encoding="utf-8"))
    manifest = {k: v for k, v in manifest.items() if v.get("gemini_verified")}
    words = {
        w["english"]: w for w in bu200["words"] if w["category"] not in EXCLUDED_CATEGORIES
    }

    rng = random.Random(SEED)
    questions = []
    pool_sizes = {}

    for lang in REFERENCE_LANGUAGES:
        pool = [
            (english, w, manifest[english]["slug"], rec)
            for english, w in words.items()
            if english in manifest
            and (rec := w.get("reference_translations", {}).get(lang))
        ]
        pool_sizes[lang] = len(pool)
        if len(pool) < N_CHOICES:
            continue  # not enough words to build a 4-way MCQ for this language

        translations = [p[3] for p in pool]
        for i, (english, w, slug, correct) in enumerate(pool):
            others = [t for j, t in enumerate(translations) if j != i and t != correct]
            if len(others) < N_CHOICES - 1:
                continue
            distractors = rng.sample(others, N_CHOICES - 1)
            choices = distractors + [correct]
            rng.shuffle(choices)
            questions.append(
                {
                    "id": f"{LANGUAGE_ISO[lang]}-{slug}",
                    "english": english,
                    "slug": slug,
                    "category": w["category"],
                    "language": lang,
                    "iso639_3": LANGUAGE_ISO[lang],
                    "choices": choices,
                    "correct_index": choices.index(correct),
                    "correct_answer": correct,
                }
            )

    out = DATA / "questions.json"
    out.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

    print("pool size per language (words with both an image and a reference translation):")
    for lang in REFERENCE_LANGUAGES:
        print(f"  {lang:10s} {pool_sizes[lang]:3d}")
    print(f"\nwrote {len(questions)} questions -> {out}")


if __name__ == "__main__":
    main()
