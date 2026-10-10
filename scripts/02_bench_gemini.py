"""Run the afrivl-bench question set against one or more Gemini models.

Each question is a single stateless `generate_content` call: the concept image
plus a lettered multiple-choice list of candidate translations in the target
language. The model answers with a single letter via a constrained JSON schema
(no chain-of-thought, no free text to parse).

Usage:
    export GEMINI_API_KEY=...
    python3 02_bench_gemini.py --models gemini-2.5-flash,gemini-2.5-pro

Resumable: appends to results/<model>.jsonl and skips question ids already
present, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from common import DATA, IMAGES, RESULTS, client

from google.genai import errors as genai_errors
from google.genai import types

LETTERS = "ABCD"
DEFAULT_MODELS = ["gemini-2.5-flash"]
CONCURRENCY_DEFAULT = 8

ANSWER_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={"answer": types.Schema(type=types.Type.STRING, enum=list(LETTERS))},
    required=["answer"],
)

_RETRYABLE = (genai_errors.APIError, asyncio.TimeoutError, ConnectionError)


def slug_model(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")


def prompt_for(q: dict) -> str:
    word = q.get("local_word", q.get("english"))
    if q.get("question_type") == "binary":
        return (
            f"The attached image is presented for the word '{word}' in {q['language']}.\n\n"
            f"Does this image correctly and accurately depict this word?\n"
            f"Answer with only 'Yes' or 'No'."
        )
    if q.get("image_options"):
        valid_letters = ", ".join(opt["letter"] for opt in q["image_options"])
        return (
            f"The images above are labeled {valid_letters} in order.\n\n"
            f"Which image correctly depicts the word '{word}' in {q['language']}?\n\n"
            f"Answer with only the letter ({valid_letters}) of the correct image."
        )
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["choices"]))
    return (
        f"The attached image shows a single concept. In {q['language']}, which of "
        f"these options is the correct word for what is pictured?\n\n{opts}\n\n"
        "Answer with only the letter of the correct option."
    )


async def ask(aio, model: str, q: dict, sem: asyncio.Semaphore) -> dict:
    parts = []
    if q.get("image_options"):
        for opt in q["image_options"]:
            img_path = DATA / opt["image_file"]
            parts.append(types.Part.from_text(text=f"Image {opt['letter']}:"))
            parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text=prompt_for(q)))
    else:
        if q.get("image_file"):
            img_path = DATA / q["image_file"]
        else:
            img_path = IMAGES / f"{q['slug']}.jpg"
        parts = [
            types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/jpeg"),
            types.Part.from_text(text=prompt_for(q)),
        ]

    contents = [types.Content(role="user", parts=parts)]

    if q.get("question_type") == "binary":
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={"answer": types.Schema(type=types.Type.STRING, enum=["Yes", "No"])},
            required=["answer"],
        )
    elif q.get("image_options"):
        valid_letters = [opt["letter"] for opt in q["image_options"]]
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={"answer": types.Schema(type=types.Type.STRING, enum=valid_letters)},
            required=["answer"],
        )
    else:
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={"answer": types.Schema(type=types.Type.STRING, enum=list(LETTERS))},
            required=["answer"],
        )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0,
    )
    async with sem:
        for attempt in range(4):
            try:
                r = await aio.models.generate_content(model=model, contents=contents, config=config)
                break
            except _RETRYABLE as e:
                if attempt == 3:
                    return {**base_row(q), "error": str(e)}
                await asyncio.sleep(2 * (attempt + 1))
    try:
        raw_ans = json.loads(r.text)["answer"].strip()
    except Exception:
        return {**base_row(q), "raw_response": getattr(r, "text", None), "error": "unparseable"}

    if q.get("question_type") == "binary":
        is_pass = (raw_ans.lower() == q.get("correct_answer", "yes").lower())
        return {
            **base_row(q),
            "model_answer": raw_ans,
            "pass": is_pass,
        }
    else:
        letter = raw_ans.upper()
        idx = LETTERS.find(letter)
        is_pass = (letter == q.get("correct_letter") or idx == q.get("correct_index"))
        return {
            **base_row(q),
            "model_answer": letter,
            "model_index": idx,
            "pass": is_pass,
        }


def base_row(q: dict) -> dict:
    return {
        "id": q["id"], "english": q["english"], "category": q["category"],
        "language": q["language"], "iso639_3": q["iso639_3"],
        "question_type": q.get("question_type", "multiple_choice"),
        "correct_answer": q.get("correct_letter", q.get("correct_answer")),
        "correct_index": q.get("correct_index", 0),
        "decision_depth": q.get("decision_depth"),
    }


async def run_model(model: str, questions: list[dict], concurrency: int):
    out_path = RESULTS / f"{slug_model(model)}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    todo = [q for q in questions if q["id"] not in done]
    print(f"[{model}] {len(done)} already done, {len(todo)} to run")
    if not todo:
        return

    c = client()
    sem = asyncio.Semaphore(concurrency)
    with open(out_path, "a", encoding="utf-8") as f:
        tasks = [asyncio.create_task(ask(c.aio, model, q, sem)) for q in todo]
        n_pass = n_done = 0
        for fut in asyncio.as_completed(tasks):
            row = await fut
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            n_pass += bool(row.get("pass"))
            if n_done % 50 == 0 or n_done == len(todo):
                print(f"  [{model}] {n_done}/{len(todo)}  running pass rate {100*n_pass/n_done:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY_DEFAULT)
    ap.add_argument("--questions", default=None, help="path to questions JSON (defaults to questions_ground_truth.json if present, else questions.json)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    args = ap.parse_args()

    if args.questions:
        q_path = Path(args.questions)
    elif (DATA / "questions_ground_truth.json").exists():
        q_path = DATA / "questions_ground_truth.json"
        print(f"Using human ground-truth benchmark questions: {q_path}")
    else:
        q_path = DATA / "questions.json"

    questions = json.loads(q_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    for model in args.models.split(","):
        asyncio.run(run_model(model.strip(), questions, args.concurrency))


if __name__ == "__main__":
    main()
