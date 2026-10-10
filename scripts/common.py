"""Shared paths, Gemini client, and constants for vl-mmlu-afri."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

# No default key: GEMINI_API_KEY must be exported. Committing a key would put a
# live credential in the repo, and a revoked one in a fallback silently turns
# every call into a bad answer rather than an error.
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Human reference translations only exist for these 9 BU 200 languages — the
# multiple-choice answer key is a human translation, not a model output, so the
# benchmark is restricted to languages the BU 200 Word Project actually covers.
REFERENCE_LANGUAGES = [
    "isiXhosa", "isiZulu", "Amharic", "Wolof", "Hausa",
    "kiSwahili", "Igbo", "Akan Twi", "Mandinka",
]
LANGUAGE_ISO = {
    "isiXhosa": "xho", "isiZulu": "zul", "Amharic": "amh", "Wolof": "wol",
    "Hausa": "hau", "kiSwahili": "swh", "Igbo": "ibo", "Akan Twi": "twi",
    "Mandinka": "mnk",
}

# This is a visual grounding benchmark over concrete nouns: real, photographable
# objects. "days of the week" has no visual referent at all. "numbers" and
# "colors" aren't objects either - a numeral or a color swatch tests symbol/
# attribute recognition, not whether the model can name a pictured thing.
EXCLUDED_CATEGORIES = {"days of the week", "numbers", "colors"}

N_CHOICES = 4
SEED = 20261010


def client():
    from google import genai

    if not API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set")
    return genai.Client(api_key=API_KEY)
