"""Aggregate every results/<model>.jsonl into results/summary.json for the dashboard."""
from __future__ import annotations

import json
from collections import defaultdict

from common import N_CHOICES, RESULTS


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def aggregate_model(rows: list[dict]) -> dict:
    scored = [r for r in rows if "pass" in r]
    errors = len(rows) - len(scored)
    n = len(scored)
    n_pass = sum(r["pass"] for r in scored)

    binary_rows = [r for r in scored if r.get("question_type") == "binary"]
    mcq_rows = [r for r in scored if r.get("question_type") != "binary"]

    by_lang = defaultdict(list)
    by_cat = defaultdict(list)
    for r in scored:
        by_lang[r["language"]].append(r)
        by_cat[r["category"]].append(r)

    return {
        "n_questions": len(rows),
        "n_scored": n,
        "n_errors": errors,
        "overall_pass_rate": pct(n_pass, n),
        "binary_pass_rate": pct(sum(r["pass"] for r in binary_rows), len(binary_rows)),
        "mcq_pass_rate": pct(sum(r["pass"] for r in mcq_rows), len(mcq_rows)),
        "chance_baseline": 25.0,
        "by_language": {
            lang: {
                "iso639_3": rs[0]["iso639_3"],
                "n": len(rs),
                "pass_rate": pct(sum(r["pass"] for r in rs), len(rs)),
            }
            for lang, rs in sorted(by_lang.items())
        },
        "by_category": {
            cat: {"n": len(rs), "pass_rate": pct(sum(r["pass"] for r in rs), len(rs))}
            for cat, rs in sorted(by_cat.items())
        },
    }


def main():
    summary = {"chance_baseline": round(100 / N_CHOICES, 1), "models": {}}
    for path in sorted(RESULTS.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        model = path.stem
        summary["models"][model] = aggregate_model(rows)
        print(f"{model:24s} n={len(rows):4d} pass_rate="
              f"{summary['models'][model]['overall_pass_rate']}%")

    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
