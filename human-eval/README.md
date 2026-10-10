---
title: VL-MMLU-Afri Human Review
emoji: 🧑‍⚖️
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: cc-by-4.0
short_description: Human review tool for the VL-MMLU-Afri image set
---

# VL-MMLU-Afri — Human Review

Pick a language, look at the picture, pick the word that matches it. Every
answer is recorded (anonymously, by a random per-browser session id) to a
public dataset, [`AfriSpeech/vl-mmlu-afri-human-eval`](https://huggingface.co/datasets/AfriSpeech/vl-mmlu-afri-human-eval),
so human accuracy can be compared against the vision-language models benchmarked
in [vl-mmlu-afri](https://github.com/AfriSpeech/vl-mmlu-afri) — not as a ground
truth to replace the model benchmark, but as a corroborating signal (and
likely a much smaller sample size).

A "this image doesn't match the word" report is a separate signal from an
answer — it flags the image itself for review rather than scoring a wrong
guess, and feeds back into fixing `data/search_hints.json` upstream.
