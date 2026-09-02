"""
ParseTune - Step 1: dataset preparation.

Turns the SROIE receipt dataset (ICDAR 2019, key-information-extraction task)
into instruction-following pairs for supervised fine-tuning of Phi-3.5-mini
with QLoRA + PEFT + TRL's SFTTrainer.

Each receipt becomes one (or more) example shaped like:

    user      -> "Extract the following fields from this receipt: {schema}. ...
                  Receipt text: {raw_text}"
    assistant -> '{"company": "...", "date": "...", ...}'

The point of ParseTune is *schema-instructable* extraction: the model must
honour whatever schema it is handed at inference time (which fields, in which
order), not just memorise one fixed output shape. So for the training split we
optionally emit extra variants of each receipt with a random subset/ordering of
the schema. Validation and test always use the canonical 4-field schema so the
metrics stay comparable across runs.

Usage
-----
    # default: pull from the Hugging Face Hub, write data/processed/*.json
    python data/prepare_data.py

    # quick smoke test on 20 receipts per split
    python data/prepare_data.py --limit 20

    # no schema augmentation (one canonical example per receipt)
    python data/prepare_data.py --schema-variants 1

    # offline: read the official SROIE files you unzipped into data/raw/
    python data/prepare_data.py --source local

Output
------
    data/processed/train.json
    data/processed/validation.json
    data/processed/test.json
    data/processed/dataset_info.json   (stats + the exact config used)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# 1. Task definition: the four SROIE key-information fields.
# --------------------------------------------------------------------------
# The description strings are what the model actually sees inside the prompt.
# They give the schema semantic meaning, which is what lets the fine-tuned
# model generalise to field names it never saw during training.
FIELD_SPECS: dict[str, str] = {
    "company": "name of the merchant or business issuing the receipt",
    "date":    "date of purchase, exactly as printed on the receipt",
    "address": "full street address of the merchant",
    "total":   "final total amount paid",
}
CANONICAL_SCHEMA: list[str] = list(FIELD_SPECS)

# Label ids used by the SROIE token-classification mirrors on the Hub.
# Order is fixed by the original `darentang/sroie` loading script; the parquet
# mirrors store bare ints, so we need this map to decode them.
BIO_LABEL_NAMES = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-ADDRESS", "I-ADDRESS",
    "B-TOTAL",   "I-TOTAL",
]

# Candidate Hub datasets, tried in order. The first is a plain-parquet mirror
# of `darentang/sroie` (626 train / 347 test, clean BIO tags) - preferred
# because it needs no remote code execution, which `datasets>=3.0` refuses.
HF_SOURCES = [
    "trng1305/sroie_dataset",
    "mp-02/sroie",
]

SYSTEM_PROMPT = (
    "You are a document extraction engine. You read a document and return only "
    "a single JSON object matching the schema you are given. No explanations."
)

# The instruction template. First line is the task; the rest pins down the
# output contract so the model's behaviour is predictable at inference time.
INSTRUCTION_TEMPLATE = """Extract the following fields from this receipt: {schema}

Respond with a single JSON object containing exactly these keys, in this order. \
If a field does not appear in the receipt, use null.

Receipt text:
{raw_text}"""


# --------------------------------------------------------------------------
# 2. Small text helpers.
# --------------------------------------------------------------------------
def clean(text: str) -> str:
    """Collapse runs of whitespace and trim. OCR output is full of them."""
    return re.sub(r"[ \t]+", " ", str(text)).strip()


def clean_block(text: str) -> str:
    """Same as `clean`, but preserves line breaks (used for receipt bodies)."""
    lines = [clean(line) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def render_schema(fields: Iterable[str]) -> str:
    """
    Render a schema as compact JSON: {"company": "name of the merchant...", ...}

    Passing descriptions (not just bare names) is deliberate - it is the
    handle the model uses to interpret unfamiliar field names later.
    """
    return json.dumps({f: FIELD_SPECS[f] for f in fields}, ensure_ascii=False)


def render_target(fields: Iterable[str], values: dict[str, str]) -> str:
    """Render the gold answer, keys in the exact order the schema asked for."""
    return json.dumps({f: values.get(f) or None for f in fields}, ensure_ascii=False)


# --------------------------------------------------------------------------
# 3. Decoding the token-classification format into field values.
# --------------------------------------------------------------------------
def tag_to_field(tag: str) -> tuple[str, str]:
    """
    Split a tag like 'B-COMPANY' into ('B', 'company').

    Handles both schemes found on the Hub: BIO ('B-'/'I-') and BIOES-style
    single tags ('S-'). Returns ('O', '') for the outside tag.
    """
    tag = str(tag)
    if tag in ("O", "o", ""):
        return "O", ""
    prefix, _, name = tag.partition("-")
    if not name:                       # tag was a bare field name, e.g. 'TOTAL'
        return "B", prefix.lower()
    return prefix.upper(), name.lower()


def extract_fields(words: list[str], tags: list[str]) -> dict[str, str]:
    """
    Walk the tagged word sequence and rebuild one string value per field.

    A new span starts when the tag prefix is 'B'/'S' *or* the field changes;
    that single rule decodes BIO and S-only schemes correctly. When a field
    has several spans (a company name repeated in a footer, say) we keep the
    longest one, which is almost always the full form rather than a fragment.

    Values are kept VERBATIM - no currency stripping, no reformatting. Checked
    against the official task-3 gold on 100 receipts, the tagged spans agree
    exactly on date 97%, total 94%, company 83%, address 60%; nearly all the
    remainder is OCR noise ('NO.5:' vs 'NO.53') or an affix ('RM11.40' vs
    '11.40'). Normalising those *lowered* agreement, because the gold is
    itself inconsistent (some totals carry three decimals). Verbatim spans are
    also the only targets actually copyable from the input text, so they are
    what a text-only model can learn. Fold the cosmetic differences into the
    comparison in evaluate/metrics.py instead, where they belong.
    """
    spans: dict[str, list[list[str]]] = {}
    prev_field = ""

    for word, tag in zip(words, tags):
        prefix, field = tag_to_field(tag)
        if not field or field not in FIELD_SPECS:
            prev_field = ""
            continue
        # 'I-' continuing the same field extends the current span; anything
        # else opens a new one.
        starts_new = prefix != "I" or field != prev_field
        bucket = spans.setdefault(field, [])
        if starts_new or not bucket:
            bucket.append([])
        bucket[-1].append(str(word))
        prev_field = field

    values: dict[str, str] = {}
    for field, bucket in spans.items():
        best = max(bucket, key=len)            # longest span wins, ties -> first
        value = clean(" ".join(best))
        if value:
            values[field] = value
    return values


# --------------------------------------------------------------------------
# 4. Rebuilding readable receipt text from OCR words + boxes.
# --------------------------------------------------------------------------
def reconstruct_text(
    words: list[str],
    bboxes: list[list[int]] | None,
    line_tol: int = 10,
    flat: bool = False,
) -> str:
    """
    Join OCR words back into a receipt-shaped block of text.

    The boxes are LayoutLM-normalised ([x0, y0, x1, y1] on a 0-1000 grid), so
    words whose vertical midpoints are within `line_tol` belong to the same
    printed line. Grouping them back into lines matters: a receipt's layout
    (header block, then address, then the total near the bottom) is a strong
    signal, and flattening everything into one long line throws it away.

    Falls back to a plain space-join when boxes are unavailable or `flat`.
    """
    if flat or not bboxes or len(bboxes) != len(words):
        return clean(" ".join(str(w) for w in words))

    # Pair each word with the vertical midpoint and left edge of its box.
    items = []
    for word, box in zip(words, bboxes):
        x0, y0, _x1, y1 = (list(box) + [0, 0, 0, 0])[:4]
        items.append((float(y0 + y1) / 2.0, float(x0), str(word)))

    items.sort(key=lambda it: (it[0], it[1]))   # top-to-bottom, then left-to-right

    lines: list[list[tuple[float, str]]] = []
    line_anchor: float | None = None
    for mid_y, x0, word in items:
        if line_anchor is None or abs(mid_y - line_anchor) > line_tol:
            lines.append([])
            line_anchor = mid_y
        lines[-1].append((x0, word))

    return clean_block(
        "\n".join(" ".join(w for _x, w in sorted(line)) for line in lines)
    )


# --------------------------------------------------------------------------
# 5. Loading: Hugging Face Hub, or the official files from data/raw/.
# --------------------------------------------------------------------------
def _decode_tag_column(dataset, column: str = "ner_tags") -> list[str] | None:
    """
    Return the id -> tag-name mapping for a token-classification column.

    Some mirrors keep the `ClassLabel` names; the parquet ones store bare
    ints, in which case we fall back to the canonical SROIE BIO ordering.
    """
    try:
        feature = dataset.features[column].feature
        names = list(getattr(feature, "names", []) or [])
        if names:
            return names
    except (KeyError, AttributeError, TypeError):
        pass
    return None


def load_from_hub(repo_id: str | None) -> dict[str, list[dict[str, Any]]]:
    """
    Load SROIE from the Hub and normalise it to a list of plain dicts:
        {"receipt_id", "raw_words", "bboxes", "tags"}
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:                                  # pragma: no cover
        raise SystemExit(
            "The `datasets` package is required for --source hf.\n"
            "  pip install datasets\n"
            "Or use --source local after unzipping SROIE into data/raw/."
        ) from exc

    candidates = [repo_id] if repo_id else HF_SOURCES
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            print(f"[load] trying Hub dataset: {candidate}")
            dsd = load_dataset(candidate)
        except Exception as exc:                                # noqa: BLE001
            print(f"[load]   failed: {type(exc).__name__}: {exc}")
            last_error = exc
            continue

        out: dict[str, list[dict[str, Any]]] = {}
        for split_name, split in dsd.items():
            names = _decode_tag_column(split) or BIO_LABEL_NAMES
            records = []
            for i, row in enumerate(split):
                # Tags may already be strings, or ids we look up in `names`.
                tags = [
                    t if isinstance(t, str) else names[int(t)]
                    for t in row.get("ner_tags", [])
                ]
                records.append({
                    "receipt_id": f"{split_name}-{row.get('id', i)}",
                    "raw_words": [str(w) for w in row.get("words", [])],
                    "bboxes": row.get("bboxes"),
                    "tags": tags,
                })
            out[split_name] = records
        print(f"[load] loaded {candidate}: "
              + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
        return out

    raise SystemExit(
        f"Could not load any SROIE dataset from the Hub ({candidates}). "
        f"Last error: {last_error}"
    )


def load_from_local(raw_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Load the official SROIE release from disk (offline fallback).

    Expects the ICDAR layout somewhere under `raw_dir`: for each receipt a
    `<name>.txt` OCR file whose lines are `x1,y1,...,x4,y4,transcription`,
    and a `<name>.json` holding the gold entities. Because these already
    carry the answers directly, no tag decoding is needed - such records
    arrive pre-resolved via the "values" / "text" keys.
    """
    json_files = sorted(p for p in raw_dir.rglob("*.json") if p.is_file())
    if not json_files:
        raise SystemExit(
            f"No .json entity files found under {raw_dir}. Unzip the SROIE "
            "task-3 data there, or use the default --source hf."
        )

    records = []
    for jf in json_files:
        try:
            entities = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(entities, dict):
            continue
        values = {
            k.lower(): clean(v)
            for k, v in entities.items()
            if k.lower() in FIELD_SPECS and isinstance(v, (str, int, float))
        }
        if not values:
            continue

        # Find the matching OCR box file and keep only the transcription
        # column (everything after the 8 coordinates).
        txt = jf.with_suffix(".txt")
        if not txt.exists():
            matches = list(raw_dir.rglob(f"{jf.stem}.txt"))
            txt = matches[0] if matches else None
        if txt is None:
            continue
        lines = []
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split(",", 8)
            if len(parts) == 9:
                lines.append(clean(parts[8]))
            elif line.strip():
                lines.append(clean(line))

        records.append({
            "receipt_id": jf.stem,
            "text": clean_block("\n".join(lines)),
            "values": values,
        })

    print(f"[load] loaded {len(records)} receipts from {raw_dir}")
    return {"all": records}


# --------------------------------------------------------------------------
# 6. Receipt -> instruction/response example(s).
# --------------------------------------------------------------------------
def sample_schemas(
    rng: random.Random,
    n_variants: int,
    available: list[str],
) -> list[list[str]]:
    """
    Build the schemas to emit for one receipt.

    Variant 0 is always the canonical full schema in canonical order, so every
    receipt teaches the primary task. Extra variants are random non-empty
    subsets in random order - that is what teaches the model to follow the
    requested field set instead of always dumping the same four keys.
    """
    schemas: list[list[str]] = [list(CANONICAL_SCHEMA)]
    seen = {tuple(CANONICAL_SCHEMA)}
    pool = available or CANONICAL_SCHEMA

    attempts = 0
    while len(schemas) < n_variants and attempts < n_variants * 10:
        attempts += 1
        size = rng.randint(1, len(pool))
        candidate = rng.sample(pool, size)
        rng.shuffle(candidate)
        if tuple(candidate) not in seen:
            seen.add(tuple(candidate))
            schemas.append(candidate)
    return schemas


def build_example(
    example_id: str,
    receipt_id: str,
    split: str,
    schema: list[str],
    raw_text: str,
    values: dict[str, str],
) -> dict[str, Any]:
    """
    Assemble one training record.

    `prompt` / `completion` are lists of chat messages - TRL calls this the
    "conversational prompt-completion" format. SFTTrainer applies the Phi-3.5
    chat template itself and masks the prompt out of the loss, so we must not
    hand-format any special tokens here. The remaining keys are metadata for
    the evaluation step (they are ignored during training).
    """
    instruction = INSTRUCTION_TEMPLATE.format(
        schema=render_schema(schema), raw_text=raw_text
    )
    return {
        "id": example_id,
        "receipt_id": receipt_id,
        "split": split,
        "schema": schema,
        "receipt_text": raw_text,
        "target": {f: values.get(f) or None for f in schema},
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "completion": [
            {"role": "assistant", "content": render_target(schema, values)},
        ],
    }


def convert_split(
    records: list[dict[str, Any]],
    split: str,
    args: argparse.Namespace,
    rng: random.Random,
    stats: Counter,
) -> list[dict[str, Any]]:
    """Convert one split's raw receipts into instruction-following examples."""
    examples: list[dict[str, Any]] = []

    for record in records:
        # Local records arrive pre-resolved; Hub records need tag decoding.
        if "values" in record:
            values = record["values"]
            raw_text = record["text"]
        else:
            values = extract_fields(record["raw_words"], record["tags"])
            raw_text = reconstruct_text(
                record["raw_words"], record.get("bboxes"),
                line_tol=args.line_tol, flat=args.flat_text,
            )

        # Drop receipts we could not read or could not label at all - they
        # would only teach the model to emit all-null objects.
        if not raw_text:
            stats["dropped_empty_text"] += 1
            continue
        if not values:
            stats["dropped_no_fields"] += 1
            continue

        if len(raw_text) > args.max_chars:
            raw_text = raw_text[: args.max_chars].rstrip()
            stats["truncated_text"] += 1

        for field in CANONICAL_SCHEMA:
            if field not in values:
                stats[f"missing_{field}"] += 1

        # Only the training split gets schema augmentation; val/test stay
        # canonical so scores are comparable run to run.
        n_variants = args.schema_variants if split == "train" else 1
        available = [f for f in CANONICAL_SCHEMA if f in values]

        for v, schema in enumerate(sample_schemas(rng, n_variants, available)):
            examples.append(build_example(
                example_id=f"sroie-{split}-{len(examples):05d}-v{v}",
                receipt_id=record["receipt_id"],
                split=split,
                schema=schema,
                raw_text=raw_text,
                values=values,
            ))

    stats[f"{split}_receipts"] += len(records)
    stats[f"{split}_examples"] += len(examples)
    return examples


# --------------------------------------------------------------------------
# 7. Splitting.
# --------------------------------------------------------------------------
def make_splits(
    raw: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """
    Produce train / validation / test.

    SROIE ships an official train/test split, and we keep it: `test` stays
    untouched so results are comparable with published numbers, and the
    validation set is carved out of `train`. When loading from local files
    (no official split) we cut all three ourselves.
    """
    if "all" in raw:                       # local: no official split
        pool = list(raw["all"])
        rng.shuffle(pool)
        n = len(pool)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)
        return {
            "test": pool[:n_test],
            "validation": pool[n_test:n_test + n_val],
            "train": pool[n_test + n_val:],
        }

    train_pool = list(raw.get("train", []))
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * args.val_ratio)

    splits = {
        "train": train_pool[n_val:],
        "validation": train_pool[:n_val],
        "test": list(raw.get("test") or raw.get("validation") or []),
    }
    # If the mirror had no test split, fall back to carving one from train.
    if not splits["test"]:
        n_test = int(len(splits["train"]) * args.test_ratio)
        splits["test"] = splits["train"][:n_test]
        splits["train"] = splits["train"][n_test:]
    return splits


# --------------------------------------------------------------------------
# 8. Entry point.
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Prepare SROIE as instruction pairs for ParseTune SFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", choices=["hf", "local"], default="hf",
                   help="Pull from the Hugging Face Hub, or read data/raw/.")
    p.add_argument("--hf-dataset", default=None,
                   help=f"Override the Hub repo id (tried in order: {HF_SOURCES}).")
    p.add_argument("--raw-dir", type=Path, default=project_root / "data" / "raw",
                   help="Where the official SROIE files live (--source local).")
    p.add_argument("--out-dir", type=Path,
                   default=project_root / "data" / "processed",
                   help="Where the processed JSON files are written.")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of the training receipts held out for validation.")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="Test fraction, used only when there is no official split.")
    p.add_argument("--schema-variants", type=int, default=2,
                   help="Examples per TRAIN receipt: 1 = canonical schema only, "
                        ">1 adds random schema subsets/orderings.")
    p.add_argument("--max-chars", type=int, default=4000,
                   help="Truncate receipt text longer than this many characters.")
    p.add_argument("--line-tol", type=int, default=10,
                   help="Vertical tolerance (0-1000 grid) for grouping OCR words "
                        "into printed lines.")
    p.add_argument("--flat-text", action="store_true",
                   help="Join OCR words with spaces instead of rebuilding lines.")
    p.add_argument("--limit", type=int, default=None,
                   help="Keep only the first N receipts per split (smoke tests).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    stats: Counter = Counter()

    # Step 1 - load the raw dataset.
    raw = (load_from_local(args.raw_dir) if args.source == "local"
           else load_from_hub(args.hf_dataset))

    # Step 2 - split into train / validation / test.
    splits = make_splits(raw, args, rng)
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    # Step 3 - convert each receipt into instruction-following pair(s).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        examples = convert_split(splits.get(split, []), split, args, rng, stats)
        path = args.out_dir / f"{split}.json"
        path.write_text(
            json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written[split] = len(examples)
        print(f"[write] {path}  ({len(splits.get(split, []))} receipts "
              f"-> {len(examples)} examples)")

    # Step 4 - record exactly how this dataset was built, next to the data.
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": FIELD_SPECS,
        "system_prompt": SYSTEM_PROMPT,
        "instruction_template": INSTRUCTION_TEMPLATE,
        "format": "TRL conversational prompt-completion (prompt/completion)",
        "counts": written,
        "stats": dict(stats),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
    }
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[done] " + "  ".join(f"{k}={v}" for k, v in written.items()))
    if stats:
        print("[stats] " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
