"""Build symmetrical benchmark questions from human-eval ground truth.

Enforces experimental fairness between human reviewers and LLMs:
Models only see candidate images up to the sequential decision point where
the human accepted an image:
- If the human accepted the 1st image: The human made a binary decision on
  that single image. The LLM is given that single image and must decide
  Yes or No.
- If the human accepted the 2nd image (after rejecting the 1st): The human's
  decision space was {image 1, image 2}. The LLM is given those exact 2 images
  (labeled A, B, shuffled) to pick between.
- If the human accepted candidate k (after rejecting candidates 1..k-1): The
  LLM is given the k images seen by the human up to that point (labeled A, B, ...,
  shuffled) to identify the correct referent.

Usage:
    python3 04_build_ground_truth_from_human_eval.py [--local-dir DIR]
Writes data/questions_ground_truth.json
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

from common import DATA, LANGUAGE_ISO, RESULTS, SEED

from huggingface_hub import HfApi, hf_hub_download

DATASET = "AfriSpeech/afrivl-bench-human-eval"
UA = "afrivl-bench/0.1 (research benchmark; contact: michseth@ghananlp.org)"
GT_IMAGES_DIR = DATA / "images" / "ground_truth"
GT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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

        # Sequential decision horizon: images seen up to point where human said Yes
        cand_idx = chosen_sub.get("candidate_index", 0)
        all_cands = chosen_sub.get("all_candidates", [])
        if not all_cands:
            all_cands = [top_img]
            cand_idx = 0

        # Exact candidates seen by the human up to that selection point
        seen_candidates = all_cands[:cand_idx + 1]

        if cand_idx == 0:
            # Case 1: 1st image was accepted -> Binary Yes/No evaluation
            correct_file = GT_IMAGES_DIR / f"{item_id}_cand0.jpg"
            ok = download_image(top_url, correct_file)
            if not ok:
                print(f"Warning: Failed to download image for {item_id}: {top_url}")
                continue

            ground_truth_questions.append({
                "id": item_id,
                "question_type": "binary",
                "language": language,
                "iso639_3": iso,
                "country": sample.get("country"),
                "english": english,
                "local_word": local_word,
                "category": category,
                "slug": slug,
                "image_file": str(correct_file.relative_to(DATA)),
                "image_url": top_url,
                "correct_answer": "Yes",
                "human_votes": n_votes,
                "total_reviews": len(subs),
                "decision_depth": 1,
            })
        else:
            # Case 2: Candidate k accepted after rejecting earlier ones -> Multiple-choice among candidates 1..k
            cand_files = []
            for c_i, c in enumerate(seen_candidates):
                dest = GT_IMAGES_DIR / f"{item_id}_cand{c_i}.jpg"
                if download_image(c["image_url"], dest):
                    cand_files.append({
                        "image_file": str(dest.relative_to(DATA)),
                        "image_url": c["image_url"],
                        "is_correct": (c_i == cand_idx),
                        "country": c.get("country"),
                    })

            if not cand_files or not any(c["is_correct"] for c in cand_files):
                print(f"Warning: Missing downloaded files for {item_id}")
                continue

            # Shuffle candidate options so the correct answer isn't predictably last
            rng.shuffle(cand_files)
            correct_idx = next(i for i, c in enumerate(cand_files) if c["is_correct"])
            correct_letter = LETTERS[correct_idx]

            image_options = [
                {
                    "letter": LETTERS[i],
                    "image_file": c["image_file"],
                    "image_url": c["image_url"],
                    "country": c.get("country"),
                }
                for i, c in enumerate(cand_files)
            ]

            ground_truth_questions.append({
                "id": item_id,
                "question_type": "multiple_choice",
                "language": language,
                "iso639_3": iso,
                "country": sample.get("country"),
                "english": english,
                "local_word": local_word,
                "category": category,
                "slug": slug,
                "image_options": image_options,
                "correct_letter": correct_letter,
                "correct_answer": correct_letter,
                "correct_index": correct_idx,
                "human_votes": n_votes,
                "total_reviews": len(subs),
                "decision_depth": cand_idx + 1,
            })

    out_file = DATA / "questions_ground_truth.json"
    out_file.write_text(json.dumps(ground_truth_questions, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write a summary of human ground truth
    by_lang_counts = Counter(q["language"] for q in ground_truth_questions)
    by_type_counts = Counter(q["question_type"] for q in ground_truth_questions)
    summary = {
        "total_human_submissions": len(submissions),
        "total_ground_truth_questions": len(ground_truth_questions),
        "none_match_items_count": len(none_match_items),
        "by_question_type": dict(by_type_counts),
        "by_language": dict(by_lang_counts),
    }
    (RESULTS / "human_ground_truth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBuilt {len(ground_truth_questions)} symmetrical benchmark questions -> {out_file}")
    print(f"Question types: {dict(by_type_counts)}")
    print(f"Items with no matching image (none match): {len(none_match_items)}")
    print(f"By language: {dict(by_lang_counts)}")


if __name__ == "__main__":
    main()
