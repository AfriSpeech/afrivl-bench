---
title: AfriVL-Bench Human Review
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: cc-by-4.0
short_description: Human ground truth for African visual vocabulary
---

# AfriVL-Bench — Human Ground Truth Review

This tool collects human visual-linguistic ground truth across 9 African languages
(Akan Twi, Hausa, Igbo, kiSwahili, Amharic, Wolof, Mandinka, isiXhosa, isiZulu).

Reviewers pick their language, view the authentic local word alongside country-grounded
candidate photos from Wikimedia Commons, and select the image that best depicts that concept
in their culture and country context (or mark if none match).

Selections are recorded to the public dataset:
[`AfriSpeech/afrivl-bench-human-eval`](https://huggingface.co/datasets/AfriSpeech/afrivl-bench-human-eval)
and serve as the ground truth against which multimodal vision-language models (e.g. Gemini) are evaluated.
