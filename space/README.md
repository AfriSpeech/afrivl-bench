---
title: VL-MMLU-Afri
emoji: 🖼️
colorFrom: purple
colorTo: green
sdk: static
pinned: false
license: cc-by-4.0
short_description: Visual multiple-choice MT benchmark, African languages
---

# VL-MMLU-Afri

Shows a vision-language model an image of a concept and asks it to pick the
correct translation, in a target African language, from 4 real human-curated
options. Tests visual grounding, not just text translation.

The app reads `results/summary.json` from the
[AfriSpeech/vl-mmlu-afri](https://github.com/AfriSpeech/vl-mmlu-afri) GitHub
repo at page load, so a new benchmark run appears here on refresh with no
redeploy.
