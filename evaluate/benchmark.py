"""
ParseTune - Step 4b: the benchmark.

Runs each system over a split (test by default), scores it with metrics.py,
and writes one timestamped JSON per run into evaluate/results/ - config,
metrics and every prediction - then rebuilds the comparison table from the
latest run of each system.

    parsetune        4-bit base + the adopted Phase 2 adapter (v2)
    zero-shot        4-bit base, bare                what the base already does
    three-shot       4-bit base, 3 worked examples   how much is formatting
    zero-shot-bf16   full-precision base, bare       how much is quantisation

Usage
-----
    # all four systems on test, logged for `tail -f`
    python -u evaluate/benchmark.py > evaluate/results/benchmark.log 2>&1

    # a quick check of the pipeline on validation
    python evaluate/benchmark.py --split validation --limit 8

    # rebuild the comparison table from saved runs
    python evaluate/benchmark.py --compare-only

Every system runs at the same batch size (default 2, recorded): batched
decoding can flip near-tied greedy choices, so systems are compared like for
like. Batch 4 ran out of GPU memory on the three-shot prompts (~3,000 tokens
each, and continuous batching holds up to twice the batch in memory while
the next prompts load); 2 peaks near 8-10GB even for the bf16 weights. A
system that fails is logged and skipped, so the others still finish. Test
is scored once per system; nothing is tuned on it afterwards.
"""

from __future__ import annotations

import os

# The bf16 weights come from the local Hugging Face cache (Phase 2 downloaded
# them); never reach for the network mid-benchmark. Must precede mlx_lm.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mlx.core as mx  # noqa: E402

from metrics import mark, summarize  # noqa: E402
from predict import (  # noqa: E402
    DATA_DIR, MODEL_PATH, PROJECT_ROOT, Progress, check_template, load_model,
    load_template, predict, project_relative, row_schema,
)

RESULTS_DIR = PROJECT_ROOT / "evaluate" / "results"
ADAPTER_PATH = PROJECT_ROOT / "train" / "adapters" / "phi35-sroie-v2"
BF16_MODEL = "microsoft/Phi-3.5-mini-instruct"       # train/config.yaml base_model

SYSTEMS: dict[str, dict[str, Any]] = {
    "parsetune":      {"model": MODEL_PATH, "adapter": "adopted", "shots": 0},
    "zero-shot":      {"model": MODEL_PATH, "adapter": None, "shots": 0},
    "three-shot":     {"model": MODEL_PATH, "adapter": None, "shots": 3},
    "zero-shot-bf16": {"model": BF16_MODEL, "adapter": None, "shots": 0},
}


# --------------------------------------------------------------------------
# 1. Pieces of a run.
# --------------------------------------------------------------------------
def pick_shots(template, n: int, seed: int):
    """
    Worked examples for the few-shot baseline: canonical four-field training
    examples, from receipts no longer than the median (so the prompt stays
    bounded), drawn with a fixed seed and recorded with the run.
    """
    rows = [r for r in json.loads((DATA_DIR / "train.json").read_text(encoding="utf-8"))
            if r["id"].endswith("-v0")]
    median = sorted(len(r["receipt_text"]) for r in rows)[len(rows) // 2]
    pool = [r for r in rows if len(r["receipt_text"]) <= median]
    chosen = random.Random(seed).sample(pool, n)
    shots = [(r["receipt_text"], dict(template.fields), r["target"]) for r in chosen]
    return shots, [r["id"] for r in chosen]


def run_system(name: str, rows: list[dict[str, Any]], template, args) -> Path:
    """Predict, score and save one system's run; returns the results file."""
    spec = SYSTEMS[name]
    adapter = args.adapter if spec["adapter"] == "adopted" else None
    shots, shot_ids = pick_shots(template, spec["shots"], args.seed) if spec["shots"] else ([], [])

    print(f"\n=== {name}: {spec['model']}" + (f" + {adapter}" if adapter else "")
          + (f", {len(shots)} shots" if shots else "")
          + f" | {len(rows)} {args.split} receipts, batch {args.batch_size} ===", flush=True)
    tic = time.perf_counter()
    model, tokenizer = load_model(spec["model"], adapter)
    print(f"[{name}] model loaded in {time.perf_counter() - tic:.0f}s", flush=True)

    tic = time.perf_counter()
    jobs = [(r["receipt_text"], row_schema(template, r)) for r in rows]
    records, stats = predict(model, tokenizer, template, jobs, args.batch_size,
                             shots, progress=Progress(name))
    seconds = time.perf_counter() - tic
    for row, record in zip(rows, records):
        record.update(id=row["id"], receipt_id=row["receipt_id"], target=row["target"])
    mark(records)
    metrics = summarize(records, list(template.fields))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out_dir / f"{stamp}_{args.split}_{name}.json"
    path.write_text(json.dumps({
        "system": name,
        "split": args.split,
        "created_utc": stamp,
        "config": {
            "model": project_relative(spec["model"]),
            "adapter": project_relative(adapter) if adapter else None,
            "shots": len(shots), "shot_ids": shot_ids, "batch_size": args.batch_size,
            "seed": args.seed, "limit": args.limit, "mlx_lm": version("mlx-lm"),
        },
        "runtime": {
            "seconds": round(seconds, 1),
            "prompt_tps": round(stats.prompt_tps, 1),
            "generation_tps": round(stats.generation_tps, 1),
            "peak_memory_gb": round(stats.peak_memory, 2),
        },
        "metrics": metrics,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    micro, ci = metrics["micro"], metrics["ci95"]
    print(f"[{name}] done in {seconds / 60:.1f} min - normalised "
          f"{100 * micro['normalised']:.1f}% [{100 * ci['normalised'][0]:.1f}-"
          f"{100 * ci['normalised'][1]:.1f}], exact {100 * micro['exact']:.1f}%, "
          f"valid JSON {100 * metrics['json_valid_raw']:.1f}% - wrote {path.name}",
          flush=True)

    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return path


# --------------------------------------------------------------------------
# 2. The comparison table.
# --------------------------------------------------------------------------
def comparison(out_dir: Path, split: str, batch_size: int) -> str:
    """
    Markdown table over the latest saved run of each system on `split` at
    `batch_size` - runs at other batch sizes never mix into one table.
    """
    latest: dict[str, Path] = {}
    for path in sorted(out_dir.glob(f"*_{split}_*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        if config["batch_size"] == batch_size:
            latest[path.stem.split("_", 2)[2]] = path      # later stamps win
    if not latest:
        return f"(no saved {split} runs at batch {batch_size} in {out_dir})"
    pct = lambda x: f"{100 * x:.1f}"                        # noqa: E731
    lines = [
        f"| System | Exact | Normalised (95% CI) | F1 | company | date | address | total "
        f"| Valid JSON (raw) | Schema adherence | Out-of-schema keys |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in SYSTEMS:
        if name not in latest:
            continue
        m = json.loads(latest[name].read_text(encoding="utf-8"))["metrics"]
        mi, ci, fields = m["micro"], m["ci95"]["normalised"], m["fields"]
        lines.append(
            f"| {name} | {pct(mi['exact'])} | {pct(mi['normalised'])} "
            f"({pct(ci[0])}–{pct(ci[1])}) | {pct(mi['f1'])} | "
            + " | ".join(pct(fields[f]["normalised"]) for f in fields)
            + f" | {pct(m['json_valid_raw'])} | {pct(m['schema_adherence'])} "
            f"| {pct(m['out_of_schema_keys'])} |")
    table = "\n".join(lines)
    (out_dir / f"comparison_{split}.md").write_text(
        f"# ParseTune — {split} split ({m['receipts']} receipts, batch {batch_size})\n\n"
        "Scores in %. Field columns are normalised match; F1 is normalised.\n\n"
        f"{table}\n", encoding="utf-8")
    return table


# --------------------------------------------------------------------------
# 3. Entry point.
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark ParseTune against its baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--systems", nargs="+", choices=list(SYSTEMS),
                   default=list(SYSTEMS), help="Which systems to run, in order.")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N receipts (pipeline checks).")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Same for every system, and recorded.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the few-shot examples and bootstrap.")
    p.add_argument("--adapter", type=Path, default=ADAPTER_PATH,
                   help="The adopted Phase 2 adapter, used by 'parsetune'.")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--compare-only", action="store_true",
                   help="Skip running; rebuild the table from saved results.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    if not args.compare_only:
        template = load_template()
        rows = json.loads((DATA_DIR / f"{args.split}.json").read_text(encoding="utf-8"))
        rows = rows[: args.limit] if args.limit else rows
        check_template(template, rows)
        print(f"[benchmark] {len(args.systems)} systems on {len(rows)} {args.split} "
              f"receipts: {', '.join(args.systems)}", flush=True)
        for name in args.systems:
            try:
                run_system(name, rows, template, args)
            except Exception:                               # noqa: BLE001
                failed.append(name)
                print(f"[{name}] FAILED - moving on to the next system\n"
                      f"{traceback.format_exc()}", flush=True)
                gc.collect()
                mx.clear_cache()

    table = comparison(args.out_dir, args.split, args.batch_size)
    print(f"\n[benchmark] comparison ({args.split}, batch {args.batch_size}):\n{table}",
          flush=True)
    print(f"[benchmark] all done" + (f" - FAILED: {', '.join(failed)}" if failed else ""),
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
