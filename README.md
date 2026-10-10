# VL-MMLU-Afri

A visual multiple-choice translation benchmark for African languages.

Each question shows an **image** of a concept and asks a model: *"In
`<language>`, which of these 4 options is the correct word for what's
pictured?"* One option is the real human-curated translation; the other three
are real human translations of *other* concepts in the same language, so every
wrong answer is still a plausible-looking word rather than invented text.

```
[image of a banana] + "In Hausa, which word means this?"
  A. ayaba      <- correct
  B. kujera
  C. mota
  D. rigar
```

This tests whether a vision-language model can ground a word in an image, not
just translate text — a model that only pattern-matches text could not answer
these questions without actually looking at the picture.

## Data

* **Concepts & reference translations** — [`data/bu200.json`](data/bu200.json),
  carried over from the
  [BU 200 Word Project](https://www.bu.edu/200word/) scrape built for
  [gemini-word-mt-bench](https://github.com/AfriSpeech/gemini-word-mt-bench),
  with human-curated reference translations into up to 9 languages (isiXhosa,
  isiZulu, Amharic, Wolof, Hausa, kiSwahili, Igbo, Akan Twi, Mandinka). Scope
  is restricted to **concrete nouns** — real, photographable objects — so
  three categories are excluded entirely: "days of the week" (no photo
  represents "Monday"), and "numbers" and "colors" (a digit or a swatch tests
  symbol/attribute recognition, not whether the model can name a pictured
  thing). The remaining 13 categories (animals, body parts, clothing, cooking,
  farm, food and drink, house, natural world, people, places, professions,
  school, transportation) are all genuinely picturable objects.
* **Images** — `scripts/00_fetch_images.py` sources one image per concept from
  Wikimedia Commons (the only source used — general web image search mostly
  surfaces paid stock photos with no redistribution rights). Wikipedia's own
  "canonical" photo for a word is reliably a generic/Western stock image blind
  to where these languages are spoken, so instead of Wikipedia's summary API
  this searches Commons directly, cycling the word with each of the 8
  countries the 9 reference languages are actually spoken in — Ghana, Nigeria,
  South Africa, Kenya, Ethiopia, Senegal, Tanzania, Gambia (from
  [afriso](https://github.com/AfriSpeech/afriso)) — e.g. `"pepper nigeria"`
  finds an actual Nigerian market instead of a seed-catalog illustration.
  "body parts" is the one exception: anatomy doesn't vary by country, and a
  country-qualified query actively hurt it (`"chin south africa"` matched a
  war-memorial statue), so that category searches the bare English word first.

  Every candidate photo — not just the first hit — is shown to Gemini (vision)
  with the question "does this actually, unambiguously depict the word?"
  before being accepted; a rejected candidate is discarded and the next one
  from the search results is tried, up to 12 candidates per word, so a bad
  keyword match (a brand name, a place name, the wrong sense of the word)
  gets caught and retried automatically rather than trusted on sight.
  `data/search_hints.json` overrides the query — or pins an exact
  `"File:..."` — for words a plain search can't resolve. Every image's
  verdict and reasoning is recorded in `data/images_manifest.json`
  (`"gemini_verified"`, `"gemini_note"`); a word that doesn't pass for any of
  its 12 candidates is simply excluded from the question set rather than
  shipped with a bad image — 13 of the 218 concrete-noun concepts (mostly
  relational/profession terms — Doctor, Father, Brother, Sibling, Goldsmith —
  that are inherently hard to photograph unambiguously) didn't clear the bar.
* **Questions** — `scripts/01_build_questions.py` builds a 4-way
  multiple-choice question for every (concept, language) pair that has both a
  verified image and a human reference translation, with 3 distractors
  sampled from other concepts' real translations into the same language.

## Pipeline

```bash
cd scripts
export GEMINI_API_KEY=...
python3 00_fetch_images.py     # → data/images/, data/images_manifest.json (search + verify, see above)
python3 01_build_questions.py  # → data/questions.json
python3 02_bench_gemini.py --models gemini-2.5-flash,gemini-2.5-pro
python3 03_aggregate.py        # → results/summary.json
```

`00_fetch_images.py` is idempotent — re-run with `--refresh` to redo words
already verified, or plain to pick up only words that still lack one (e.g.
after adding a `search_hints.json` entry). `scripts/00b_verify_images.py` is a
standalone re-check tool for spot-auditing the manifest later; the main
pipeline no longer needs it since verification is now inline in step 1.

`02_bench_gemini.py` is resumable per model — it appends to
`results/<model>.jsonl` and skips question ids already answered.

```bash
python3 04_aggregate_human_eval.py   # pulls human-eval/ submissions → merges into results/summary.json
```

## Current dataset size

**205** of **218** concrete-noun concepts have a verified image (13 excluded
for lacking one — see above) → **1,207** questions across the 9 reference
languages:

| language | questions |
|---|---|
| Igbo | 156 |
| isiXhosa / isiZulu | 143 each |
| Hausa | 137 |
| kiSwahili | 134 |
| Wolof | 127 |
| Amharic / Mandinka | 123 each |
| Akan Twi | 121 |

## Results

First full run, all 1,207 questions, two Gemini generations:

| model | overall pass rate |
|---|---|
| gemini-3.6-flash | **94.8%** |
| gemini-2.5-flash | **90.3%** |

gemini-3.6-flash beats gemini-2.5-flash on **every single category** — a
clean generational improvement, not noise. Hardest categories for both models
are school, professions, clothing and farm (85–92%); easiest are food and
drink, animals and transportation (96–99%), against a 25% chance baseline.
Full per-language/per-category breakdown is in
[`results/summary.json`](results/summary.json) and the
[dashboard](https://huggingface.co/spaces/AfriSpeech/vl-mmlu-afri).

## Models

Currently benchmarked via the Gemini API only (`google-genai` SDK,
`GEMINI_API_KEY` env var). Local/open-weight vision-language models may be
added later as a second runner.

## Dashboard

A static HTML/JS page in [`space/`](space/), deployed as a static Hugging Face
Space, reads `results/summary.json` from this repo at page load — the same
pattern as
[gemini-afro-mt-bench](https://github.com/AfriSpeech/gemini-afro-mt-bench).

## Human review

A second static page, [`human-eval/`](human-eval/), lets a person step through
the same image-and-4-choices questions by hand. Every answer (and every
"this image doesn't match the word" report) is written client-side straight
to the public dataset
[`AfriSpeech/vl-mmlu-afri-human-eval`](https://huggingface.co/datasets/AfriSpeech/vl-mmlu-afri-human-eval)
via a Hugging Face **write-only, single-dataset-scoped** fine-grained token
embedded in the page — there's no backend to hide it behind on a static page,
so the token is deliberately limited to "can only write to this one dataset"
so the worst case is spam, not a real security issue.

`scripts/04_aggregate_human_eval.py` pulls every submission from that dataset
and merges it into `results/summary.json` as a `"human-eval"` entry —
structurally identical to a model's entry, so it shows up as just another row
in the same leaderboard and detail views the dashboard already has, no UI
changes needed. This is deliberately **not treated as ground truth**: human
sample size will likely stay far smaller than the per-model LLM runs, so it's
a corroborating signal (does the model's accuracy track human accuracy,
per language/category?) rather than a replacement answer key. Image-quality
reports land in `results/human_flags.json` for manual review, feeding back
into `data/search_hints.json` upstream.

## Caveats

* The multiple-choice format means a model can get partial credit from
  elimination/guessing; chance baseline is 25% (4 options).
* Distractors are drawn from the *same* language's other concepts, so a model
  that recognizes "this is a body-part word in Hausa" without knowing which
  body part could still do better than chance on categories where concepts
  within a category share obvious lexical patterns. Category-level pass rates
  should be read against the 25% baseline, not against 0%.
* Reference translations exist for only 9 languages (the ones the BU 200 Word
  Project itself published), so this benchmark's language coverage is far
  smaller than the word-level round-trip benchmark's 1,622 languages — it is
  a different, narrower measurement (visual grounding) on a fixed set of
  "anchor" languages with a human-verified answer key.
