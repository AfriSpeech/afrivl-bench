"""Build visual-choice benchmark questions from human-eval ground truth.

Pulls human ground-truth submissions from AfriSpeech/vl-mmlu-afri-human-eval
(or a local directory) written by human-eval/index.html under
`ground_truth/<session_id>/<id>-<ts>.json`.

Evaluation task symmetry:
Just as humans were shown candidate images alongside the local word to pick
the matching image, the LLM is evaluated on the exact same task:
Given 4 images labeled A, B, C, D (the human-selected image + 3 visual distractors
from the same candidate pool) and the local word in that language, the model must
pick the letter corresponding to the correct image.

Usage:
    python3 04_build_ground_truth_from_human_eval.py [--local-dir DIR]
Writes data/questions_ground_truth.json:
    [ { "id": "...", "language": "...", "local_word": "...", "english": "...",
        "image_options": [ {"letter": "A", "image_file": "...", "image_url": "..."}, ...],
        "correct_letter": "B", "correct_index": 1, ... } ]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

from common import DATA, LANGUAGE_ISO, N_CHOICES, RESULTS, SEED

from huggingface_hub import HfApi, hf_hub_download

DATASET = "AfriSpeech/vl-mmlu-afri-human-eval"
UA = "vl-mmlu-afri/0.1 (research benchmark; contact: michseth@ghananlp.org)"
GT_IMAGES_DIR = DATA / "images" / "ground_truth"
GT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCD"


def download_image(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                dest.write_bytes(r.read())
                return True
        except Exception:
            if attempt == 2:
                return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", type=str, default=None, help="Read submissions from local dir instead of HF")
    args = ap.parse_args()

    submissions = []
    if args.local_dir:
        p = Path(args.local_dir)
        for f in p.glob("**/*.json"):
            try:
                submissions.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        print(f"Loaded {len(submissions)} submissions from local directory {p}")
    else:
        api = HfApi()
        try:
            files = api.list_repo_files(DATASET, repo_type="dataset")
            gt_files = [f for f in files if f.startswith("ground_truth/") and f.endswith(".json")]
            print(f"Found {len(gt_files)} ground-truth submissions in {DATASET}")
            for path in gt_files:
                local = hf_hub_download(DATASET, path, repo_type="dataset")
                submissions.append(json.loads(open(local, encoding="utf-8").read()))
        except Exception as e:
            print(f"Could not fetch from HF dataset: {e}", file=sys.stderr)

    if not submissions:
        print("No ground-truth submissions found yet.")
        print(f"Reviewers can submit selections via the human-eval space.")
        return

    # Group submissions by item_id: e.g. "twi-airplane"
    by_item = defaultdict(list)
    for sub in submissions:
        if sub.get("type") == "ground_truth_selection":
            item_id = sub.get("id")
            if item_id:
                by_item[item_id].append(sub)

    print(f"Total unique concept items with human review: {len(by_item)}")
    rng = random.Random(SEED)

    ground_truth_questions = []
    none_match_items = []
    skipped_items = []

    for item_id, subs in by_item.items():
        sample = subs[0]
        language = sample["language"]
        english = sample["english"]
        local_word = sample["local_word"]
        category = sample["category"]
        slug = sample["slug"]
        iso = sample.get("iso639_3", LANGUAGE_ISO.get(language, "unk"))

        actions = Counter(s.get("action") for s in subs)
        n_none = actions.get("no_match_exhausted", 0) + actions.get("none_match", 0)
        n_match = actions.get("matched", 0) + actions.get("selected", 0)
        if n_none > n_match:
            none_match_items.append({"id": item_id, "language": language, "english": english, "votes": actions})
            continue

        selected_subs = [s for s in subs if s.get("action") in ("matched", "selected") and s.get("selected_image")]
        if not selected_subs:
            skipped_items.append({"id": item_id, "language": language, "english": english})
            continue

        # Vote on best image
        img_votes = Counter(s["selected_image"]["image_url"] for s in selected_subs)
        top_url, n_votes = img_votes.most_common(1)[0]
        chosen_sub = next(s for s in selected_subs if s["selected_image"]["image_url"] == top_url)
        top_img = chosen_sub["selected_image"]

        # Collect candidate pool from submissions
        all_cands = []
        seen_cand_urls = set()
        for s in subs:
            for c in s.get("all_candidates", []):
                u = c.get("image_url")
                if u and u not in seen_cand_urls:
                    seen_cand_urls.add(u)
                    all_cands.append(c)

        # Distractor images are other candidate images from the same pool that were not the chosen image
        distractor_cands = [c for c in all_cands if c.get("image_url") != top_url]
        if len(distractor_cands) < N_CHOICES - 1:
            # If candidate pool had fewer than 3 distractors, keep whatever is available
            distractors = distractor_cands
        else:
            distractors = rng.sample(distractor_cands, N_CHOICES - 1)

        # Download correct image
        correct_file = GT_IMAGES_DIR / f"{item_id}_correct.jpg"
        ok = download_image(top_url, correct_file)
        if not ok:
            print(f"Warning: Failed to download correct image for {item_id}: {top_url}")
            continue

        # Download distractor images
        distractor_entries = []
        for d_i, d in enumerate(distractors):
            d_dest = GT_IMAGES_DIR / f"{item_id}_distractor_{d_i}.jpg"
            if download_image(d["image_url"], d_dest):
                distractor_entries.append({
                    "image_file": str(d_dest.relative_to(DATA)),
                    "image_url": d["image_url"],
                    "is_correct": False,
                    "country": d.get("country"),
                })

        if len(distractor_entries) < 1:
            print(f"Warning: Not enough distractors downloaded for {item_id}")
            continue

        # Build multiple choice image options (shuffled)
        correct_entry = {
            "image_file": str(correct_file.relative_to(DATA)),
            "image_url": top_url,
            "is_correct": True,
            "country": top_img.get("country"),
            "artist": top_img.get("artist"),
            "license": top_img.get("license"),
        }
        all_options = distractor_entries + [correct_entry]
        rng.shuffle(all_options)

        correct_idx = next(i for i, opt in enumerate(all_options) if opt["is_correct"])
        correct_letter = LETTERS[correct_idx]

        image_options = [
            {
                "letter": LETTERS[i],
                "image_file": opt["image_file"],
                "image_url": opt["image_url"],
                "country": opt.get("country"),
            }
            for i, opt in enumerate(all_options)
        ]

        ground_truth_questions.append({
            "id": item_id,
            "language": language,
            "iso639_3": iso,
            "country": sample.get("country"),
            "english": english,
            "local_word": local_word,
            "category": category,
            "slug": slug,
            "image_options": image_options,
            "correct_letter": correct_letter,
            "correct_index": correct_idx,
            "human_votes": n_votes,
            "total_reviews": len(subs),
            "top_image_url": top_url,
        })

    out_file = DATA / "questions_ground_truth.json"
    out_file.write_text(json.dumps(ground_truth_questions, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write a summary of human ground truth
    by_lang_counts = Counter(q["language"] for q in ground_truth_questions)
    summary = {
        "total_human_submissions": len(submissions),
        "total_ground_truth_questions": len(ground_truth_questions),
        "none_match_items_count": len(none_match_items),
        "by_language": dict(by_lang_counts),
    }
    (RESULTS / "human_ground_truth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBuilt {len(ground_truth_questions)} visual-choice benchmark questions -> {out_file}")
    print(f"Items with no matching image (none match): {len(none_match_items)}")
    print(f"By language: {dict(by_lang_counts)}")


if __name__ == "__main__":
    main()
