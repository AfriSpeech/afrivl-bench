"""Pull human-eval submissions from the HF dataset and merge into results/summary.json.

Human answers are written as `responses/<session>/<qid>-<ts>.json` to the
AfriSpeech/vl-mmlu-afri-human-eval dataset by human-eval/index.html. This
fetches every response file, aggregates it exactly like 03_aggregate.py
aggregates a model's run (same shape: overall_pass_rate, by_language,
by_category), and writes it into results/summary.json under the key
"human-eval" so the dashboard shows it as just another row in the leaderboard
-- side by side with the models, not as ground truth, and the row's own
n_scored makes the (likely small) sample size visible rather than hidden.

"flag" submissions (a human reporting the image doesn't match the word) are
summarized separately into results/human_flags.json for manual review; they
are not scored answers.

Usage:
    python3 04_aggregate_human_eval.py
"""
from __future__ import annotations

import json
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download

from common import RESULTS

DATASET = "AfriSpeech/vl-mmlu-afri-human-eval"


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def main():
    api = HfApi()
    files = api.list_repo_files(DATASET, repo_type="dataset")

    responses, flags = [], []
    for path in files:
        if path.startswith("responses/") and path.endswith(".json"):
            local = hf_hub_download(DATASET, path, repo_type="dataset")
            responses.append(json.loads(open(local, encoding="utf-8").read()))
        elif path.startswith("flags/") and path.endswith(".json"):
            local = hf_hub_download(DATASET, path, repo_type="dataset")
            flags.append(json.loads(open(local, encoding="utf-8").read()))

    print(f"fetched {len(responses)} answers, {len(flags)} image-flags")

    if responses:
        by_lang, by_cat = defaultdict(list), defaultdict(list)
        for r in responses:
            by_lang[r["language"]].append(r)
            by_cat[r["category"]].append(r)

        human_entry = {
            "n_questions": len(responses),
            "n_scored": len(responses),
            "n_errors": 0,
            "overall_pass_rate": pct(sum(r["human_correct"] for r in responses), len(responses)),
            "chance_baseline": 25.0,
            "by_language": {
                lang: {
                    "iso639_3": rs[0]["iso639_3"], "n": len(rs),
                    "pass_rate": pct(sum(r["human_correct"] for r in rs), len(rs)),
                }
                for lang, rs in sorted(by_lang.items())
            },
            "by_category": {
                cat: {"n": len(rs), "pass_rate": pct(sum(r["human_correct"] for r in rs), len(rs))}
                for cat, rs in sorted(by_cat.items())
            },
        }

        summary_path = RESULTS / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {
            "chance_baseline": 25.0, "models": {}
        }
        summary["models"]["human-eval"] = human_entry
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"human-eval: n={human_entry['n_scored']} pass_rate={human_entry['overall_pass_rate']}%")
        print(f"wrote into {summary_path}")
    else:
        print("no human responses yet - summary.json left unchanged")

    if flags:
        by_word = defaultdict(list)
        for f in flags:
            by_word[f["english"]].append(f)
        flags_out = {
            "n_flags": len(flags),
            "by_word": {
                w: {"n": len(fs), "notes": [f.get("note") for f in fs if f.get("note")]}
                for w, fs in sorted(by_word.items(), key=lambda kv: -len(kv[1]))
            },
        }
        (RESULTS / "human_flags.json").write_text(
            json.dumps(flags_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n{len(by_word)} distinct words flagged as bad images:")
        for w, v in flags_out["by_word"].items():
            print(f"  {w:20s} flagged {v['n']}x  {v['notes']}")


if __name__ == "__main__":
    main()
