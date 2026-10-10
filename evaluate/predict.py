"""
ParseTune - Step 3: inference.

Loads the 4-bit base model - with a Phase 2 adapter, or bare for the Phase 4
baselines - renders a receipt and any schema through the exact template the
training data was built with, and decodes greedily in batches. Every output
keeps its raw text next to the parsed object, and anything the JSON repair
pass had to fix is flagged rather than hidden, so Phase 4 can report raw
validity honestly.

Usage
-----
    # validation split, trained adapter
    python evaluate/predict.py --split validation --adapter train/adapters/phi35-sroie

    # bare base model on the first 20 test receipts, predictions saved
    python evaluate/predict.py --split test --limit 20 --out base-test.json

    # one receipt, any schema (field -> description)
    python evaluate/predict.py --adapter train/adapters/phi35-sroie \
        --receipt receipt.txt --schema '{"merchant_name": "name of the shop"}'

Library use (Phase 4 and 5 import these):
    load_template, build_messages, load_model, predict, extract_json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from mlx_lm import load
from mlx_lm.generate import BatchGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "phi-3.5-mini-instruct-4bit"


def project_relative(path: Any) -> str:
    """
    A path as saved files record it: relative to the repository when it lies
    inside it, unchanged otherwise (a Hub id, a directory elsewhere) - so a
    results file never carries the machine it was produced on.
    """
    p = Path(path)
    if p.is_absolute() and p.is_relative_to(PROJECT_ROOT):
        return p.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


# --------------------------------------------------------------------------
# 1. The prompt: exactly what the model was trained on.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Template:
    system_prompt: str
    instruction: str
    max_chars: int
    fields: dict[str, str]          # the four SROIE fields and their descriptions


def load_template(data_dir: Path = DATA_DIR) -> Template:
    """
    Read the prompt contract Phase 1 recorded next to the data it built.

    Taking the template from dataset_info.json rather than restating it here
    means inference can never drift from training: if the data is rebuilt
    with a different template, inference follows automatically.
    """
    info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    return Template(
        system_prompt=info["system_prompt"],
        instruction=info["instruction_template"],
        max_chars=info["config"]["max_chars"],
        fields=info["fields"],
    )


def build_messages(
    template: Template,
    receipt_text: str,
    schema: dict[str, str],
    shots: Iterable[tuple[str, dict[str, str], dict[str, Any]]] = (),
) -> list[dict[str, str]]:
    """
    Chat messages for one extraction: system prompt, optional demonstrations,
    then the query.

    `schema` maps each requested field to its description. The description
    is not optional: it is what lets the model interpret a field name it
    never trained on. Rendering matches Phase 1 exactly - same json.dumps
    settings, same template, same truncation - and `check_template` proves it
    against the stored prompts. `shots` are (receipt_text, schema, answer)
    triples for the few-shot baseline; each becomes a user/assistant pair.
    """
    def user_turn(text: str, fields: dict[str, str]) -> dict[str, str]:
        if len(text) > template.max_chars:             # Phase 1's truncation
            text = text[: template.max_chars].rstrip()
        return {
            "role": "user",
            "content": template.instruction.format(
                schema=json.dumps(fields, ensure_ascii=False), raw_text=text
            ),
        }

    messages = [{"role": "system", "content": template.system_prompt}]
    for text, fields, answer in shots:
        messages.append(user_turn(text, fields))
        messages.append({"role": "assistant",
                         "content": json.dumps(answer, ensure_ascii=False)})
    messages.append(user_turn(receipt_text, schema))
    return messages


def row_schema(template: Template, row: dict[str, Any]) -> dict[str, str]:
    """
    A Phase 1 row's schema with its descriptions: rows asking for fields
    beyond SROIE's four carry their own `schema_spec`; the rest use the four.
    """
    return row.get("schema_spec") or {f: template.fields[f] for f in row["schema"]}


def check_template(template: Template, rows: list[dict[str, Any]]) -> None:
    """Rebuild every stored prompt from its parts; any difference is a bug."""
    for row in rows:
        rebuilt = build_messages(template, row["receipt_text"],
                                 row_schema(template, row))
        if rebuilt != row["prompt"]:
            raise SystemExit(f"[template] prompt for {row['id']} differs from "
                             "the one Phase 1 trained on - not predicting.")


# --------------------------------------------------------------------------
# 2. Finding and repairing the JSON answer.
# --------------------------------------------------------------------------
def scan_object(text: str) -> tuple[int, int | None, int, bool]:
    """
    Locate the first top-level JSON object in `text`.

    Returns (start, end, depth, in_string): `end` is the index of its closing
    brace, or None if the object never closes - in which case `depth` and
    `in_string` describe where the text stopped. Braces inside quoted values
    do not count. `start` is -1 when there is no "{" at all.
    """
    start = text.find("{")
    if start < 0:
        return -1, None, 0, False
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i, 0, False
    return start, None, depth, in_string


def extract_json(text: str) -> tuple[dict[str, Any] | None, bool]:
    """
    Parse the model's answer, repairing what can be repaired - and saying so.

    Returns (object, repaired). A clean answer parses as-is. Otherwise the
    first top-level {...} is taken, which strips markdown fences and cuts
    any prose around it; an object left open by the token limit is closed
    (open string ended, dangling comma dropped, braces balanced). `repaired`
    is True whenever any of that was needed, so raw validity can be reported
    separately. Returns (None, True) when nothing usable is left.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, False
    except json.JSONDecodeError:
        pass

    start, end, depth, in_string = scan_object(text)
    if start < 0:
        return None, True
    if end is not None:
        candidate = text[start:end + 1]
    else:
        candidate = text[start:] + ('"' if in_string else "")
        candidate = candidate.rstrip().rstrip(",") + "}" * depth
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, True
    return (parsed, True) if isinstance(parsed, dict) else (None, True)


def token_budget(schema: dict[str, str]) -> int:
    """
    Room for the answer: 80 tokens per field plus braces and slack.

    Measured on the corpus, the longest gold answer for every schema size
    fits in under two-thirds of its budget (worst case: one long address,
    78 of 128), so the limit only bites on a model that has wandered off -
    which the finish reason ("length") then records.
    """
    return 48 + 80 * len(schema)


# --------------------------------------------------------------------------
# 3. Generation.
# --------------------------------------------------------------------------
def load_model(model_path: Path = MODEL_PATH, adapter_path: Path | None = None):
    """The 4-bit base, with the Phase 2 adapter on top when one is given."""
    return load(str(model_path),
                adapter_path=str(adapter_path) if adapter_path else None)


def generate_batch(
    model,
    tokenizer,
    message_lists: list[list[dict[str, str]]],
    max_tokens: list[int],
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[tuple[str, str, int]], Any]:
    """
    Greedy-decode many prompts with mlx-lm's continuous batching.

    Each sequence ends at Phi-3.5's end-of-turn token, at its token budget,
    or as soon as its first JSON object closes - whichever comes first - so a
    chatty baseline does not spend its budget on prose after the answer.
    Returns (text, finish_reason, n_tokens) per prompt, in order, plus the
    generator's throughput stats. `progress(done, total)` is called each
    time a sequence finishes.

    `batch_size` bounds memory: Phi-3.5 has no grouped-query attention, so
    its KV cache costs ~0.4MB per token, and prefill activations come on top
    - batch 8 peaked at 10.5GB on validation receipts. Never run this beside
    a training run: on a 24GB Mac the two together exhausted GPU memory.
    """
    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True)
               for m in message_lists]
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        prefill_batch_size=batch_size,
        completion_batch_size=batch_size,
    )
    uids = gen.insert(prompts, max_tokens)
    tokens: dict[int, list[int]] = {uid: [] for uid in uids}
    finish: dict[int, str] = {}

    with gen.stats() as stats:
        while responses := gen.next_generated():
            closed = []
            finished_before = len(finish)
            for r in responses:
                if r.finish_reason == "stop":           # end-of-turn token
                    finish[r.uid] = "eos"
                    continue
                tokens[r.uid].append(r.token)
                if r.finish_reason == "length":
                    finish[r.uid] = "length"
                elif ("}" in tokenizer.decode([r.token])
                      and scan_object(tokenizer.decode(tokens[r.uid]))[1]
                      is not None):
                    finish[r.uid] = "brace"
                    closed.append(r.uid)
            if closed:
                gen.remove(closed)
            if progress and len(finish) > finished_before:
                progress(len(finish), len(uids))
    gen.close()

    results = [(tokenizer.decode(tokens[uid]), finish.get(uid, "eos"),
                len(tokens[uid])) for uid in uids]
    return results, stats


def predict(
    model,
    tokenizer,
    template: Template,
    jobs: list[tuple[str, dict[str, str]]],
    batch_size: int = 8,
    shots: Iterable[tuple[str, dict[str, str], dict[str, Any]]] = (),
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """
    Extract every (receipt_text, schema) job and return one record per job.

    Each record keeps the raw text, whether it was valid JSON as generated,
    the parsed object (after repair, if any), the repaired flag, and why
    generation stopped. Scoring is Phase 4's job.
    """
    shots = list(shots)
    messages = [build_messages(template, text, schema, shots)
                for text, schema in jobs]
    budgets = [token_budget(schema) for _text, schema in jobs]
    outputs, stats = generate_batch(model, tokenizer, messages, budgets,
                                    batch_size, progress)
    records = []
    for (raw, reason, n_tokens), (_text, schema) in zip(outputs, jobs):
        parsed, repaired = extract_json(raw)
        records.append({
            "schema": list(schema),
            "raw": raw,
            "valid_raw": not repaired,
            "parsed": parsed,
            "repaired": repaired and parsed is not None,
            "finish": reason,
            "n_tokens": n_tokens,
        })
    return records, stats


class Progress:
    """
    A `progress` callback for predict(): one line every ~5% of jobs, with
    rate and ETA, flushed at once - made for following a run with `tail -f`.
    """

    def __init__(self, name: str, unit: str = "receipts") -> None:
        self.name, self.unit = name, unit
        self.start, self.next = time.perf_counter(), 1

    def __call__(self, done: int, total: int) -> None:
        if done < self.next and done < total:
            return
        self.next = done + max(1, total // 20)
        elapsed = time.perf_counter() - self.start
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate / 60 if rate else 0.0
        print(f"[{self.name}] {done}/{total} {self.unit} ({100 * done / total:.0f}%)"
              f" - {rate:.2f}/s - about {eta:.1f} min left", flush=True)


# --------------------------------------------------------------------------
# 4. Entry point.
# --------------------------------------------------------------------------
def summarize(records: list[dict[str, Any]], stats, seconds: float) -> None:
    """One-screen health check of a run; proper metrics live in Phase 4."""
    n = len(records)
    valid = sum(r["valid_raw"] for r in records)
    repaired = sum(r["repaired"] for r in records)
    failed = sum(r["parsed"] is None for r in records)
    print(f"[predict] {n} receipts in {seconds:.1f}s - valid JSON as "
          f"generated {valid}/{n}, repaired {repaired}, unusable {failed}")
    print(f"[predict] finish reasons: {dict(Counter(r['finish'] for r in records))}")
    print(f"[predict] prompt {stats.prompt_tps:.0f} tok/s, generation "
          f"{stats.generation_tps:.0f} tok/s, peak memory "
          f"{stats.peak_memory:.1f} GB")
    scored = [r for r in records if "target" in r]
    if scored:
        exact = sum(r["parsed"] is not None and r["parsed"].get(k) == v
                    for r in scored for k, v in r["target"].items())
        total = sum(len(r["target"]) for r in scored)
        print(f"[predict] spot check: {exact}/{total} fields exactly match "
              "the gold value (verbatim)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ParseTune extraction over a split or one receipt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=["train", "validation", "test"],
                     help="A Phase 1 split from data/processed/.")
    src.add_argument("--receipt", type=Path,
                     help="A text file holding one receipt.")
    p.add_argument("--schema", type=json.loads, default=None,
                   help='For --receipt: {"field": "description", ...}. '
                        "Defaults to the four SROIE fields.")
    p.add_argument("--adapter", type=Path, default=None,
                   help="LoRA adapter directory; omit for the bare base model.")
    p.add_argument("--model", type=Path, default=MODEL_PATH,
                   help="The MLX 4-bit base model.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N rows of the split.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Receipts decoded concurrently (bounds KV memory).")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the prediction records to this JSON file.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = load_template()

    if args.split:
        rows = json.loads((DATA_DIR / f"{args.split}.json").read_text(encoding="utf-8"))
        rows = rows[: args.limit] if args.limit else rows
        check_template(template, rows)
        jobs = [(r["receipt_text"], row_schema(template, r)) for r in rows]
    else:
        schema = args.schema or template.fields
        rows = [{"id": str(args.receipt), "schema": list(schema)}]
        jobs = [(args.receipt.read_text(encoding="utf-8"), schema)]

    model, tokenizer = load_model(args.model, args.adapter)
    tic = time.perf_counter()
    records, stats = predict(model, tokenizer, template, jobs, args.batch_size)
    seconds = time.perf_counter() - tic

    for row, record in zip(rows, records):
        record["id"] = row["id"]
        if "target" in row:
            record["target"] = row["target"]

    if args.receipt:
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
    summarize(records, stats, seconds)
    if args.out:
        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[predict] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
