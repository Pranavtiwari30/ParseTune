"""
ParseTune - Step 2: QLoRA fine-tune of Phi-3.5-mini on Apple Silicon (MLX).

Trains a LoRA adapter on top of a frozen 4-bit MLX copy of
microsoft/Phi-3.5-mini-instruct, using the instruction pairs Phase 1 wrote to
data/processed/. Everything runs locally through mlx-lm, Apple's LLM toolkit
for MLX, so this needs an Apple Silicon Mac (built on an M5 Pro, 24GB).

One command runs the whole phase:

    1. convert the base model to MLX 4-bit, once (models/, ~2GB);
    2. export the Phase 1 splits as mlx-lm chat JSONL (train/data/);
    3. check the loss mask on every example - the tokens the loss sees must
       decode to the assistant's JSON answer and nothing else;
    4. train, recording the train/validation loss curve;
    5. reload the saved adapter from disk and extract one validation receipt.

Usage
-----
    # full run: `epochs` passes over the Phase 1 training split
    python train/sft_train.py

    # smoke test: 20 examples per split, 5 iterations, separate adapter dir
    python train/sft_train.py --limit 20 --iters 5

    # steps 1-3 only, no training
    python train/sft_train.py --check-only

Output
------
    <adapter_path>/adapters.safetensors   (the LoRA weights - the deliverable)
    <adapter_path>/adapter_config.json    (every mlx-lm setting used)
    <adapter_path>/run_log.json           (loss curve, speed, memory, sample)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import yaml
from mlx_lm import generate, load
from mlx_lm.convert import convert
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Phase 1 split name -> mlx-lm file name. Test is deliberately absent: it
# belongs to the evaluation phases, and nothing in training may touch it.
SPLITS = {"train": "train", "validation": "valid"}


# --------------------------------------------------------------------------
# 1. Configuration.
# --------------------------------------------------------------------------
PATH_KEYS = ("model_path", "data_dir", "export_dir", "adapter_path",
             "smoke_adapter_path")


def load_config(path: Path) -> dict[str, Any]:
    """Read train/config.yaml, resolving its paths against the project root."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        cfg[key] = PROJECT_ROOT / cfg[key]
    return cfg


def resolve_mlx_options(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    n_train: int,
) -> dict[str, Any]:
    """
    Build the complete option set for mlx-lm's LoRA trainer.

    Starts from mlx-lm's own defaults, puts config.yaml's `mlx_lm` block on
    top, then fills in what depends on the run: paths, the iteration count,
    one checkpoint per epoch, and the learning-rate schedule. mlx-lm counts
    `iters` in micro-batches but steps the schedule once per optimizer update
    (every `grad_accumulation_steps` micro-batches), so warmup and decay are
    sized in updates - otherwise the cosine would end a quarter of the way in.
    """
    opts = {**CONFIG_DEFAULTS, **cfg["mlx_lm"]}
    steps_per_epoch = n_train // opts["batch_size"]
    iters = args.iters or steps_per_epoch * cfg["epochs"]
    updates = max(iters // opts["grad_accumulation_steps"], 1)
    warmup = round(updates * cfg["warmup_ratio"])
    adapter_path = cfg["smoke_adapter_path"] if args.limit else cfg["adapter_path"]

    opts.update(
        model=str(cfg["model_path"]),
        data=str(cfg["export_dir"]),
        adapter_path=str(adapter_path),
        train=True,
        test=False,
        iters=iters,
        save_every=max(steps_per_epoch, 1),
        lr_schedule={
            "name": "cosine_decay",
            "arguments": [opts["learning_rate"], max(updates - warmup, 1), 0.0],
            "warmup": warmup,
            "warmup_init": 0.0,
        },
    )
    if args.limit:                     # smoke runs are short: report every step
        opts["steps_per_report"] = 1
    return opts


# --------------------------------------------------------------------------
# 2. The frozen base: an MLX 4-bit copy of Phi-3.5-mini, made once.
# --------------------------------------------------------------------------
def ensure_base_model(cfg: dict[str, Any]) -> Path:
    """
    Convert the Hugging Face checkpoint to MLX 4-bit unless it already exists.

    Quantising the frozen base is what makes this QLoRA: the 3.8B weights sit
    in about 2GB while only the small adapter trains in full precision.
    Converting here, rather than pulling a community upload, pins the weights
    to Microsoft's release and the quantisation to what config.yaml says.
    """
    model_path = cfg["model_path"]
    if (model_path / "config.json").exists():
        return model_path
    if model_path.exists():
        raise SystemExit(
            f"{model_path} exists but holds no converted model (an interrupted "
            "conversion?). Delete it and run again."
        )

    q = cfg["quantization"]
    print(f"[model] converting {cfg['base_model']} -> {model_path} "
          f"({q['bits']}-bit, group size {q['group_size']}); "
          "one-time download of ~7.6GB")
    convert(
        hf_path=cfg["base_model"],
        mlx_path=str(model_path),
        quantize=True,
        q_bits=q["bits"],
        q_group_size=q["group_size"],
    )
    return model_path


# --------------------------------------------------------------------------
# 3. Phase 1 output -> mlx-lm chat JSONL.
# --------------------------------------------------------------------------
def export_chat_jsonl(
    cfg: dict[str, Any],
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Write each split as one {"messages": [system, user, assistant]} per line.

    Only `messages` is written, and that matters: mlx-lm picks the dataset
    format from the keys it finds, and a record still carrying Phase 1's
    `prompt` / `completion` lists would be read as the plain-string
    completions format instead of chat. Phase 1's metadata (target, schema,
    receipt_id) stays in data/processed/; the original rows are returned so
    later steps can use it.
    """
    cfg["export_dir"].mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for src, dst in SPLITS.items():
        rows = json.loads(
            (cfg["data_dir"] / f"{src}.json").read_text(encoding="utf-8")
        )
        if limit:
            rows = rows[:limit]
        path = cfg["export_dir"] / f"{dst}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                record = {"messages": row["prompt"] + row["completion"]}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        rows_by_split[dst] = rows
        print(f"[export] {path}  ({len(rows)} examples)")
    return rows_by_split


# --------------------------------------------------------------------------
# 4. The loss-mask check.
# --------------------------------------------------------------------------
def check_loss_mask(datasets: dict[str, Any], tokenizer, max_seq_length: int):
    """
    Confirm, for every example, that the loss sees the answer and nothing else.

    mlx-lm's `mask_prompt` keeps only the final (assistant) message in the
    loss. If the mask slipped into the prompt, the model would spend its
    capacity learning to reproduce receipts instead of extracting from them,
    and nothing downstream would flag it. So each example goes through
    mlx-lm's own dataset code - the exact tokens and mask offset training
    uses - and the span past the offset is decoded. Stripped of special
    tokens it must equal the JSON answer exactly: any prompt text that leaked
    in, or any answer text masked out, breaks the equality.

    (mlx-lm 0.32's default_loss also counts one padding target after the
    final <|endoftext|> - `steps <= lengths` in its mask. It sits past the
    <|end|> that generation stops at, so it is noted here, not failed on.)
    """
    failures, total, longest, shown = 0, 0, 0, False
    for name, ds in datasets.items():
        for i in range(len(ds)):
            item = ds[i]
            tokens, offset = ds.process(item)
            longest = max(longest, len(tokens))
            total += 1
            span = tokens[offset:]
            answer = item["messages"][-1]["content"]
            if tokenizer.decode(span, skip_special_tokens=True).strip() != answer:
                failures += 1
                if failures <= 3:
                    print(f"[mask] FAIL {name}[{i}]: "
                          f"{tokenizer.decode(span)[:160]!r}")
            elif not shown:
                print(f"[mask] trained span of {name}[{i}], special tokens "
                      f"included: {tokenizer.decode(span)!r}")
                shown = True

    print(f"[mask] {total - failures}/{total} examples train on the answer "
          f"only; longest example {longest} tokens "
          f"(max_seq_length {max_seq_length})")
    if failures:
        raise SystemExit(f"[mask] {failures} bad loss masks - not training.")
    if longest > max_seq_length:
        print(f"[warn] examples over {max_seq_length} tokens will be truncated")


def warn_skipped_tails(datasets: dict[str, Any], batch_size: int) -> None:
    """mlx-lm never uses examples that cannot fill a batch; say so if any."""
    for name, ds in datasets.items():
        if skipped := len(ds) % batch_size:
            print(f"[warn] {name}: the {skipped} longest of {len(ds)} examples "
                  f"never fill a batch of {batch_size}; mlx-lm skips them")


# --------------------------------------------------------------------------
# 5. Training.
# --------------------------------------------------------------------------
class LossLog(TrainingCallback):
    """Keeps every report mlx-lm makes, so the run can be judged afterwards."""

    def __init__(self) -> None:
        self.train: list[dict[str, float]] = []
        self.val: list[dict[str, float]] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        self.train.append({k: float(v) for k, v in train_info.items()})

    def on_val_loss_report(self, val_info: dict) -> None:
        self.val.append({k: float(v) for k, v in val_info.items()})


def train(ns: SimpleNamespace, model, train_set, valid_set) -> LossLog:
    """
    Run mlx-lm's LoRA trainer with a loss logger attached.

    `train_model` is called directly rather than through the mlx_lm.lora CLI
    because the CLI's entry point replaces any callback it is given, and the
    loss curve is what the exit criteria are judged on.
    """
    log = LossLog()
    np.random.seed(ns.seed)            # mlx-lm draws batch order from numpy
    train_model(ns, model, train_set, valid_set, training_callback=log)
    return log


# --------------------------------------------------------------------------
# 6. Round trip: reload the adapter and extract one receipt.
# --------------------------------------------------------------------------
def verify_adapter(ns: SimpleNamespace, example: dict[str, Any]) -> dict[str, Any]:
    """
    Reload base + saved adapter from disk and extract one validation receipt.

    This is the round trip Phase 3 depends on: the adapter must load back
    onto the 4-bit base, and greedy decoding (mlx-lm's default sampler) must
    produce a JSON object. Scoring proper is Phase 4's job; this only reports
    whether the output parses and how many fields match.
    """
    model, tokenizer = load(ns.model, adapter_path=ns.adapter_path)
    prompt = tokenizer.apply_chat_template(
        example["prompt"], add_generation_prompt=True, return_dict=False
    )
    output = generate(model, tokenizer, prompt=prompt, max_tokens=256)
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    is_obj = isinstance(parsed, dict)
    target = example["target"]
    return {
        "receipt_id": example["receipt_id"],
        "output": output,
        "parses": is_obj,
        "keys_match": is_obj and list(parsed) == example["schema"],
        "fields_exact": sum(is_obj and parsed.get(k) == v
                            for k, v in target.items()),
        "fields_total": len(target),
    }


def summarize(log: LossLog, sample: dict[str, Any], adapter_path: Path) -> None:
    """Print the Phase 2 exit criteria and save the run log by the adapter."""
    train_losses = [r["train_loss"] for r in log.train]
    val_losses = [r["val_loss"] for r in log.val]

    print("\n[exit criteria]")
    if len(train_losses) > 1:
        went_down = train_losses[-1] < train_losses[0]
        print(f"  train loss     {train_losses[0]:.3f} -> {train_losses[-1]:.3f}  "
              f"({'decreased' if went_down else 'did NOT decrease'})")
    if val_losses:
        best = min(val_losses)
        steady = val_losses[-1] <= best * 1.10
        print(f"  val loss       {val_losses[0]:.3f} -> {val_losses[-1]:.3f}, "
              f"best {best:.3f}  "
              f"({'no divergence' if steady else 'rose >10% above its best'})")
    print(f"  adapter        saved and reloaded from {adapter_path}")
    print(f"  sample JSON    {'parses' if sample['parses'] else 'does NOT parse'}; "
          f"keys {'match' if sample['keys_match'] else 'do NOT match'} the "
          f"schema; {sample['fields_exact']}/{sample['fields_total']} fields exact")
    print(f"  output         {sample['output']!r}")
    if log.train:
        # mlx-lm's tokens/s counts only tokens inside the loss mask - the
        # answers - so iterations/s is the honest speed figure.
        it_s = np.median([r["iterations_per_second"] for r in log.train])
        tok_s = np.median([r["tokens_per_second"] for r in log.train])
        peak = max(r["peak_memory"] for r in log.train)
        print(f"  speed / memory {it_s:.2f} it/s ({tok_s:.0f} answer tokens/s), "
              f"peak {peak:.1f} GB")

    (adapter_path / "run_log.json").write_text(
        json.dumps({"train": log.train, "val": log.val, "sample": sample},
                   indent=2),
        encoding="utf-8",
    )


def record_relative_paths(adapter_path: Path) -> None:
    """
    Rewrite the paths in mlx-lm's adapter_config.json relative to the project.

    mlx-lm saves its options exactly as it received them, and it receives
    absolute paths so a run never depends on the working directory. The saved
    copy is a record, though - loading reads only the LoRA settings from it -
    and a record should not carry the machine it was written on.
    """
    path = adapter_path / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model", "data", "adapter_path"):
        value = Path(config[key])
        if value.is_absolute() and value.is_relative_to(PROJECT_ROOT):
            config[key] = value.relative_to(PROJECT_ROOT).as_posix()
    path.write_text(json.dumps(config, indent=4), encoding="utf-8")


# --------------------------------------------------------------------------
# 7. Entry point.
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune of Phi-3.5-mini with MLX for ParseTune.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "train" / "config.yaml",
                   help="Training configuration.")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N examples per split (smoke "
                        "tests); the adapter goes to smoke_adapter_path.")
    p.add_argument("--iters", type=int, default=None,
                   help="Override the iteration count derived from `epochs`.")
    p.add_argument("--check-only", action="store_true",
                   help="Convert, export and check the loss mask; no training.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    # Step 1 - the frozen 4-bit base model.
    ensure_base_model(cfg)

    # Step 2 - Phase 1 output -> mlx-lm chat JSONL.
    rows = export_chat_jsonl(cfg, args.limit)
    ns = SimpleNamespace(**resolve_mlx_options(cfg, args, len(rows["train"])))

    # Step 3 - load exactly what training will see, and check the loss mask.
    model, tokenizer = load(ns.model, lazy=args.check_only)
    train_set, valid_set, _ = load_dataset(ns, tokenizer)
    datasets = {"train": train_set, "valid": valid_set}
    check_loss_mask(datasets, tokenizer, ns.max_seq_length)
    warn_skipped_tails(datasets, ns.batch_size)
    if args.check_only:
        return 0

    # Step 4 - train.
    log = train(ns, model, train_set, valid_set)
    del model
    mx.clear_cache()

    # Step 5 - reload the adapter from disk and extract one validation receipt.
    sample = verify_adapter(ns, rows["valid"][0])
    summarize(log, sample, Path(ns.adapter_path))
    record_relative_paths(Path(ns.adapter_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
