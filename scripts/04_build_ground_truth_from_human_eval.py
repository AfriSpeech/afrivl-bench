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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

from common import DATA, LANGUAGE_ISO, RESULTS, SEED

from huggingface_hub import snapshot_download

DATASET = "AfriSpeech/afrivl-bench-human-eval"
UA = "afrivl-bench/0.1 (research benchmark; contact: michseth@ghananlp.org)"
GT_IMAGES_DIR = DATA / "images" / "ground_truth"
GT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def download_one(task: tuple[str, Path]) -> tuple[str, bool]:
    url, dest = task
    if dest.exists() and dest.stat().st_size > 1000:
        return url, True
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                dest.write_bytes(r.read())
                return url, True
        except Exception:
            if attempt == 2:
                return url, False
    return url, False


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
        print(f"Syncing submissions from Hugging Face dataset {DATASET}...")
        try:
            local_dir = snapshot_download(DATASET, repo_type="dataset", allow_patterns="ground_truth/**/*.json")
            for f in Path(local_dir).glob("ground_truth/**/*.json"):
                try:
                    submissions.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
            print(f"Found {len(submissions)} ground-truth submissions in {DATASET}")
        except Exception as e:
            print(f"Could not fetch snapshot from HF: {e}", file=sys.stderr)

    if not submissions:
        print("No ground-truth submissions found yet.")
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

    # 1. First pass: analyze items and collect images that must be downloaded
    planned_items = []
    download_tasks = {}  # url -> dest_path
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

        cand_idx = chosen_sub.get("candidate_index", 0)
        all_cands = chosen_sub.get("all_candidates", [])
        if not all_cands:
            all_cands = [top_img]
            cand_idx = 0

        seen_candidates = all_cands[:cand_idx + 1]

        if cand_idx == 0:
            correct_file = GT_IMAGES_DIR / f"{item_id}_cand0.jpg"
            download_tasks[top_url] = correct_file
            planned_items.append({
                "type": "binary", "item_id": item_id, "language": language, "iso": iso,
                "country": sample.get("country"), "english": english, "local_word": local_word,
                "category": category, "slug": slug, "file": correct_file, "url": top_url,
                "votes": n_votes, "total_reviews": len(subs),
            })
        else:
            cand_entries = []
            for c_i, c in enumerate(seen_candidates):
                dest = GT_IMAGES_DIR / f"{item_id}_cand{c_i}.jpg"
                download_tasks[c["image_url"]] = dest
                cand_entries.append({
                    "dest": dest, "url": c["image_url"], "is_correct": (c_i == cand_idx),
                    "country": c.get("country"),
                })
            planned_items.append({
                "type": "multiple_choice", "item_id": item_id, "language": language, "iso": iso,
                "country": sample.get("country"), "english": english, "local_word": local_word,
                "category": category, "slug": slug, "cands": cand_entries,
                "votes": n_votes, "total_reviews": len(subs), "cand_idx": cand_idx,
            })

    # 2. Download all required images in parallel
    print(f"Downloading {len(download_tasks)} unique candidate images in parallel...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(download_one, (u, p)) for u, p in download_tasks.items()]
        completed = 0
        for fut in as_completed(futures):
            completed += 1
            if completed % 15 == 0 or completed == len(download_tasks):
                print(f"  [{completed}/{len(download_tasks)}] images cached")

    # 3. Build benchmark questions
    ground_truth_questions = []
    for item in planned_items:
        item_id = item["item_id"]
        if item["type"] == "binary":
            if not item["file"].exists() or item["file"].stat().st_size < 500:
                continue
            ground_truth_questions.append({
                "id": item_id,
                "question_type": "binary",
                "language": item["language"],
                "iso639_3": item["iso"],
                "country": item["country"],
                "english": item["english"],
                "local_word": item["local_word"],
                "category": item["category"],
                "slug": item["slug"],
                "image_file": str(item["file"].relative_to(DATA)),
                "image_url": item["url"],
                "correct_answer": "Yes",
                "human_votes": item["votes"],
                "total_reviews": item["total_reviews"],
                "decision_depth": 1,
            })
        else:
            valid_cands = [
                c for c in item["cands"]
                if c["dest"].exists() and c["dest"].stat().st_size > 500
            ]
            if not any(c["is_correct"] for c in valid_cands):
                continue

            # Shuffle candidate options
            shuffled = valid_cands[:]
            rng.shuffle(shuffled)
            correct_idx = next(i for i, c in enumerate(shuffled) if c["is_correct"])
            correct_letter = LETTERS[correct_idx]

            image_options = [
                {
                    "letter": LETTERS[i],
                    "image_file": str(c["dest"].relative_to(DATA)),
                    "image_url": c["url"],
                    "country": c.get("country"),
                }
                for i, c in enumerate(shuffled)
            ]

            ground_truth_questions.append({
                "id": item_id,
                "question_type": "multiple_choice",
                "language": item["language"],
                "iso639_3": item["iso"],
                "country": item["country"],
                "english": item["english"],
                "local_word": item["local_word"],
                "category": item["category"],
                "slug": item["slug"],
                "image_options": image_options,
                "correct_letter": correct_letter,
                "correct_answer": correct_letter,
                "correct_index": correct_idx,
                "human_votes": item["votes"],
                "total_reviews": item["total_reviews"],
                "decision_depth": item["cand_idx"] + 1,
            })

    out_file = DATA / "questions_ground_truth.json"
    out_file.write_text(json.dumps(ground_truth_questions, indent=2, ensure_ascii=False), encoding="utf-8")

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
    print(f"By language: {dict(by_lang_counts)}")


if __name__ == "__main__":
    main()
