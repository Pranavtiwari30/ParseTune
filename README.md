# ParseTune

**Schema-instructable document extraction.** A fine-tune of Phi-3.5-mini (3.8B) that reads a
document and returns *exactly the JSON schema you hand it at inference time* — the fields you
asked for, in the order you asked for them, and nothing else.

Most document-extraction models learn one fixed output shape. ParseTune learns the *instruction*:
the schema travels in the prompt, so a model trained on receipts responds to field names and
orderings it never saw during training. Trained with QLoRA on Apple's MLX (`mlx-lm`) on the
SROIE receipt corpus, locally on an Apple Silicon MacBook (M5 Pro, 24GB unified memory).

```
Input   ->  Extract the following fields from this receipt:
            {"company": "name of the merchant...", "total": "final total amount paid"}
            Receipt text: SANYU STATIONERY SHOP / NO. 31G&33G, JALAN SETIA INDAH ...

Output  ->  {"company": "SANYU STATIONERY SHOP", "total": "4.5000"}
```

---

## Results at a glance

On SROIE's official test split (347 receipts, never used for any choice), normalised match with
95% intervals — the full tables, per-field precision / recall / F1 included, are in
[`evaluate/results/REPORT.md`](evaluate/results/REPORT.md):

| System | Normalised | Exact | F1 | Valid JSON |
|:--|--:|--:|--:|--:|
| **ParseTune (v2 adapter)** | **94.2** (92.9–95.5) | **93.7** | **94.4** | **99.7** |
| Phi-3.5-mini, zero-shot, full precision | 68.0 (66.0–70.2) | 55.0 | 68.0 | 100.0 |
| Phi-3.5-mini, 3-shot, 4-bit | 54.3 (51.5–56.8) | 41.4 | 55.2 | 96.5 |
| Phi-3.5-mini, zero-shot, 4-bit | 44.6 (41.7–47.6) | 33.6 | 47.4 | 88.2 |

**Schema-following on names it never trained on** (Phase 5, test): renamed fields 96.1% right
(base model: 77.8%), a random subset in random order 94.1% with the requested keys kept every time
(base: 65.9%, adding keys nobody asked for), and an absent field (`shipping_tracking_number`)
answered `null` on 345 of 345 receipts. **The limit:** on concepts it was never taught, the
fine-tune is *worse* than the base model it started from — GST id 90% vs 99% recall, payment
method 74% vs 99% — because it copies nearby text ("Cashier") where the base answers the concept.
The instruction generalises across names and shapes, not to new concepts.

How the project got here — every decision and why the changed ones changed — is in
[`decisions.md`](decisions.md); the phases, architecture and open items are in
[`memory.md`](memory.md).

---

## Status at a glance

| Phase | What it delivers | State |
|:--|:--|:--|
| **1 — Data** | SROIE to instruction pairs, split, saved as JSON | **Done & verified** |
| **2 — Training** | QLoRA SFT of Phi-3.5-mini with MLX | **Done & verified** |
| **3 — Inference** | Constrained generation + JSON repair | **Done & verified** |
| **4 — Evaluation** | Metrics, baselines, benchmark harness | **Done & verified** — 94.2% on test |
| **5 — Generalization** | The schema-instructability test | **Done & verified** — 4 held-out conditions |
| **6 — Packaging** | Notebook, adapter card, report, write-up | **Done** — Hub upload left to the owner |

---

## Repository layout

```
ParseTune/
├── data/
│   ├── raw/                     # optional: official SROIE files, for offline runs
│   ├── processed/               # [done] train/validation/test.json + dataset_info.json
│   ├── prepare_data.py          # [done] Phase 1
│   └── requirements.txt         # [done] Phase 1 dependencies
├── train/
│   ├── sft_train.py             # [done] Phase 2 — convert, export, mask check, train, verify
│   ├── config.yaml              # [done] Phase 2
│   ├── requirements.txt         # [done] Phase 2 dependencies (mlx-lm, pinned)
│   ├── data/                    # Phase 2 — mlx-lm chat JSONL, rebuilt every run (git-ignored)
│   └── adapters/                # Phase 2 — LoRA adapters (weights git-ignored; run logs kept)
├── models/                      # Phase 2 — MLX 4-bit base model, made once (git-ignored)
├── evaluate/
│   ├── predict.py               # [done] Phase 3 — loading + generation, shared by Phases 4–5
│   ├── benchmark.py             # [done] Phase 4 — runs every system over a split
│   ├── metrics.py               # [done] Phase 4 — scoring
│   ├── schema_probe.py          # [done] adapter selection (validation) + Phase 5 (test)
│   ├── report.py                # [done] Phase 6 — results/REPORT.md from the saved runs
│   └── results/                 # [done] every run's JSON, REPORT.md, comparison tables
├── notebooks/
│   ├── parsetune.ipynb          # [done] Phase 6 — end-to-end walkthrough, executed
│   └── requirements.txt         # [done] notebook-only dependencies
├── tests/
│   └── test_pipeline.py         # regression tests: metrics, JSON repair, prompts, data
├── MODEL_CARD.md                # [done] Phase 6 — adapter card + how to publish it
├── decisions.md                 # every decision, and why the changed ones changed
├── memory.md                    # phases, architecture, scope, artifacts, open items
└── README.md
```

---

## Phase 1 — Data preparation `[DONE]`

**Deliverable:** `data/prepare_data.py` -> `data/processed/*.json`

Loads SROIE (ICDAR 2019, key-information-extraction task) from the Hugging Face Hub, with an
offline fallback that reads the official release out of `data/raw/`. Four fields:
`company`, `date`, `address`, `total`.

**What the script does**

1. **Load.** Tries parquet mirrors of `darentang/sroie` in order — no remote code execution, which
   `datasets>=3.0` refuses to run anyway.
2. **Decode labels.** Walks the BIO/BIOES token tags back into one string value per field. When a
   field has several spans (a company name repeated in the footer), the longest wins.
3. **Rebuild the receipt.** OCR words carry LayoutLM-normalised boxes, so words whose vertical
   midpoints fall within a tolerance are regrouped into printed lines. Layout is signal — the total
   sits near the bottom, the address under the header — and flattening to one long line discards it.
4. **Emit instruction pairs** in TRL's *conversational prompt-completion* format.
5. **Split** at the receipt level, then convert — so augmented variants of one receipt can never
   straddle two splits. Before converting, any train or validation receipt whose text also appears
   in test is dropped: SROIE's official split repeats 7 test receipts in train under different ids,
   so only a text comparison catches them.
6. **Add absent-field examples** to train — real receipts asked for fields they do not all carry,
   so `null` is something the model has actually seen (below).

**Design decisions worth knowing**

- **Schema carries descriptions, not just names.** The prompt gets
  `{"company": "name of the merchant or business issuing the receipt", ...}`. That description is
  the handle the model uses to interpret an unfamiliar field name at inference time. It is the
  whole mechanism behind Phase 5.
- **Schema augmentation on train only.** Each training receipt yields the canonical 4-field example
  plus extra variants with random field subsets in random order (63 distinct schemas across the
  train split). Validation and test stay canonical so scores are comparable run to run.
- **Messages, not pre-rendered strings.** `prompt`/`completion` are chat-message lists. Phase 2
  joins them into mlx-lm chat records; mlx-lm applies the Phi-3.5 chat template itself and masks
  the prompt out of the loss, so hand-formatting special tokens here would double them.
- **Values are verbatim OCR spans.** `"total": "4.5000"`, not `4.50`. Checked against the official
  task-3 gold on 100 receipts, verbatim spans agree exactly on date 97%, total 94%, company 83%,
  address 60% — and *normalising lowered that*, because the gold is itself inconsistent. Verbatim
  spans are also the only targets actually copyable from the input, so they are what a text-only
  model can learn. Cosmetic differences get folded into the comparison in Phase 4, where they belong.
- **Real nulls, from real receipts (train only).** SROIE's four fields are on nearly every receipt,
  so the four-field data holds 1 `null` answer in 1,114 examples — and a model that never sees one
  fills every field with something (asked for a shipping tracking number, the first adapter
  returned a document number off the receipt). So each training receipt gets one more example
  mixing 1–3 of its SROIE fields with fields beyond them. Nothing in it is generated: the receipt
  is real, and every answer is either a verbatim span of it or a `null` whose absence was checked.
  - *Never-present fields* (`flight_number`, `patient_name`, `vehicle_plate_number`,
    `hotel_room_number`, `bank_account_number`, `student_id`) cannot occur on these retail
    receipts; a receipt carrying any of their keywords is skipped.
  - *Grounding fields* (`time_of_purchase`, `store_phone`, `store_fax`, `store_email`) are printed
    on some receipts and not others, so the same name sometimes has a value and sometimes `null` —
    the model has to read the receipt rather than learn "unfamiliar name → null". A strict pattern
    labels a field only when it finds exactly one value, or rules the field out; anything in
    between is skipped, not guessed. Picks are balanced so each field appears with values and
    with nulls as evenly as the receipts allow.
  - The result: 557 examples carrying 495 `null` answers (9% of all training slots) and 251 real
    values. A hand audit of 60 values and 40 nulls found one systematic flaw — OCR splits some
    email addresses (`IVAN KONG12@…`) — fixed by requiring an address to start its line or follow
    a label.
  - Phase 5's held-out names and concepts (tracking/shipping/courier, payment, GST, line items)
    are refused by an assertion, and a third pool of absent fields is reserved for choosing
    between adapters (`evaluate/schema_probe.py`) and never trained on.

**Output**

| Split | Receipts | Examples | Schemas |
|:--|--:|--:|--:|
| train | 557 | 1,671 | 63 four-field + 557 mixed |
| validation | 62 | 62 | canonical only |
| test | 347 | 347 | canonical only |

Official SROIE test split kept untouched; validation carved from train at 10%; the 7 train
receipts that duplicate test receipts are dropped.

**Verified:** every `completion` parses as JSON and matches `target`, with key order equal to
`schema` (0 inconsistencies over 2,080 examples); every grounding value is a verbatim span of its
receipt, and every never-present field is `null` on a receipt free of its keywords; 0 duplicate
ids; 0 empty receipt texts; 0 receipts shared between any two splits, by id or by receipt text;
1 receipt missing a field corpus-wide. `--absent-variants 0` reproduces the four-field data byte
for byte.

```bash
python data/prepare_data.py                      # default: Hub -> data/processed/
python data/prepare_data.py --limit 20           # smoke test
python data/prepare_data.py --schema-variants 1  # canonical schema only
python data/prepare_data.py --absent-variants 0  # without the absent-field examples
python data/prepare_data.py --source local       # offline, from data/raw/
```

`data/processed/dataset_info.json` records the exact template, field specs, counts, and full
argument set used for the run sitting next to it.

---

## Phase 2 — Training `[DONE]`

**Deliverable:** `train/sft_train.py`, `train/config.yaml`, `train/requirements.txt`

QLoRA fine-tune of `microsoft/Phi-3.5-mini-instruct` on Apple Silicon with MLX (`mlx-lm`). The
base weights are converted once to MLX 4-bit and stay frozen; only a small LoRA adapter trains.

```bash
python train/sft_train.py                        # full run
python train/sft_train.py --limit 20 --iters 5   # smoke test, separate adapter directory
python train/sft_train.py --check-only           # convert + export + mask check, no training
```

One command runs the phase: convert the base model (first run only), export the Phase 1 splits as
mlx-lm chat JSONL, check the loss mask on every example, train, then reload the saved adapter from
disk and extract one validation receipt. Only train and validation are exported — test never gets
near training.

**Configuration** (`config.yaml`; where it differs, the original CUDA plan is in brackets)

| Knob | Value | Why |
|:--|:--|:--|
| Quantisation | MLX 4-bit affine, group size 64 [NF4, double-quant] | QLoRA on Apple Silicon; bitsandbytes' NF4 path is built for CUDA |
| LoRA rank / scale / dropout | 16 / 2.0 / 0.05 [alpha 32] | mlx-lm's `scale` multiplies the LoRA output directly, so 2.0 is alpha 32 / rank 16 |
| Target modules | `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, all 32 blocks | Phi-3 uses *fused* projections — the usual `q_proj`/`k_proj` names silently match nothing. mlx-lm adapts only the last 16 blocks unless told otherwise |
| Max sequence length | 2048 | A cap, never reached: the longest example is 1,178 tokens |
| Batch / grad-accum | 2 x 4 [1 x 8] | Effective batch 8. mlx-lm batches by length and silently skips the tail that cannot fill a batch — always the longest examples. Batch 2 scores all 62 validation examples and leaves out only the single longest of the 1,671 training examples; batch 8 would drop 7 and never score 6 of the 62 |
| LR / schedule / epochs | 2e-4, cosine, 3% warmup, 3 | 2,505 micro-batches = 626 optimizer updates (v1: 1,671 = 417); warmup and decay are sized in updates, which is what the schedule counts |
| Optimizer | AdamW, no weight decay [`paged_adamw_8bit`] | 8-bit and paged optimizers are CUDA tools; unified memory has the headroom without them |
| Gradient checkpointing | on | Measured on the smoke run: off is ~30% faster but peaks at 16.9 GB on 780-token batches (4.7 GB on) — too close to 24GB for the 1,178-token ones |

**Loss masking is the thing to get right.** The model should learn only from the assistant turn.
If the loss leaks into the prompt, it spends its capacity learning to reproduce receipts instead
of extracting from them. mlx-lm's `mask_prompt` does the masking, and `sft_train.py` checks it on
every example before training: each one goes through mlx-lm's own dataset code, and the tokens past
the mask offset, stripped of special tokens, must equal the JSON answer exactly. Training will not
start otherwise. On the current data all 1,733 train and validation examples pass, and the trained
span reads `{"company": …, "total": "4.5000"}<|end|><|endoftext|>`. (mlx-lm 0.32's loss also
counts one padding target after `<|endoftext|>` — an off-by-one in its mask that sits past the
`<|end|>` where generation stops, so it never reaches the output.)

**Smoke run** (`--limit 20 --iters 5`, M5 Pro): end to end in under a minute. 25.17M trainable
parameters — 0.66% of 3.82B, i.e. rank 16 on all four projections in all 32 blocks, as intended;
about 0.4 iterations/s at batch 2; peak memory 4.7 GB. The adapter saves, reloads from disk, and
its greedy output parses as JSON with exactly the requested keys (the values are not expected to
be right after one optimizer update). At that speed the full 1,671-iteration run takes roughly 75
minutes.

**Full run, v1** (four-field data; `train/adapters/phi35-sroie`): 1,671 iterations, 3 epochs, 76
minutes on the M5 Pro, peak memory 5.6 GB. Train loss 0.570 → 0.005. Validation loss went from
0.626 before training to 0.019 by iteration 700 and held there through epochs 2 and 3 (one blip to
0.022 at iteration 1,100); the final value, 0.019, is also the best.

**Full run, v2 — the adopted adapter** (with Phase 1's absent-field examples;
`train/adapters/phi35-sroie-v2`): 2,505 iterations over 1,671 examples, 3 epochs, 2 h 15 min,
peak memory 5.6 GB. Train loss 0.579 → under 0.001. Validation loss 0.626 → 0.021, best 0.019 —
the same floor as v1 — with no divergence. The training set is odd-sized, so mlx-lm's batching
leaves out its single longest example each epoch; that receipt still trains through its other
variants. v2 replaced v1 only after clearing the adoption bar agreed before retraining
(`evaluate/schema_probe.py`, validation only — results under Phase 5).

Both runs keep one checkpoint per epoch beside the final adapter, and the full loss curve in
`run_log.json`. On the final check's validation receipt, both reloaded adapters return parseable
JSON with 3 of 4 fields exact.

**Exit criteria — all met, by both runs:** training loss decreases and validation loss does not
diverge; a smoke run on `--limit 20` completes end-to-end; the adapter saves and reloads; one
greedy generation on a validation receipt produces parseable JSON. `sft_train.py` prints each check
at the end of a run.

---

## Phase 3 — Inference `[DONE]`

**Deliverable:** `evaluate/predict.py` (loading + generation path shared by Phases 4 and 5)

```bash
python evaluate/predict.py --split validation --adapter train/adapters/phi35-sroie-v2
python evaluate/predict.py --split test --limit 20 --out base-test.json      # bare base model
python evaluate/predict.py --adapter train/adapters/phi35-sroie-v2 \
    --receipt receipt.txt --schema '{"merchant_name": "name of the shop"}'   # any schema
```

Loads the 4-bit base — with the Phase 2 adapter, or bare for the baselines — renders a receipt and
any schema (`{field: description}`) through the prompt template Phase 1 used, and decodes.

- **The same prompt as training, provably.** The template is read from `dataset_info.json` rather
  than restated, and before predicting on a split every stored prompt is rebuilt from its parts —
  all 2,080 match byte for byte, including the absent-field examples that carry their own field
  descriptions. A mismatch stops the run.
- **Greedy decoding** (mlx-lm's default sampler) — extraction has one right answer, sampling only
  invents.
- **Three stop conditions:** Phi-3.5's end-of-turn token; a token budget of 48 + 80 per field (the
  longest gold answer of any schema size uses at most ~60% of it); or the first JSON object
  closing, so a chatty model cannot spend its budget on prose after the answer. Each record notes
  which one fired.
- **A JSON repair pass:** take the first top-level `{...}` — which drops markdown fences and any
  prose around it — and close an object the budget cut off. Recorded as a *repaired* flag beside
  the raw text, never hidden, so Phase 4 can report raw JSON validity honestly.
- **Batched generation** on mlx-lm's `BatchGenerator` (continuous batching): the 62 validation
  receipts take about a minute at batch 8, peaking at 9–12 GB.

**Verified on the validation split** (a spot check — the real evaluation is Phase 4, on test):

| | ParseTune adapter (v1 and v2 alike) | Bare 4-bit base |
|:--|--:|--:|
| Valid JSON as generated | 62/62 | 54/62 |
| Unusable outputs | 0 | 8 |
| Fields exactly right (verbatim) | 234/248 (94%) | 83/248 (33%) |

v1 and v2 score the same here but miss different fields: v2 fixes two totals and a truncated
company name that v1 got wrong, and gets two receipts' store identity wrong instead (a promotional
header, and a fuel brand in place of the operating company). About half of either adapter's 14
misses are noise in the gold spans rather than model errors — gold
`Date:11/03/18` against a predicted `11/03/18`, or gold addresses that begin with a fragment of the
line above. Phase 4's normalised match and per-example inspection have to tell the two apart.

**Batch size can change outputs.** Batched and single-sequence decoding do not produce bit-identical
logits, so near-tied greedy choices can flip: with the bare base model, 5 of 16 validation outputs
differed between batch 8 and batch 1. With the adapter all 16 were identical, and matched
mlx-lm's unbatched `generate`. Phase 4 fixes the batch size and records it with every run.

---

## Phase 4 — Evaluation `[DONE]`

**Deliverable:** `evaluate/metrics.py`, `evaluate/benchmark.py`, `evaluate/results/*.json`

```bash
# all four systems on test, ~36 min of compute; keep the Mac awake for the whole run
caffeinate -i python -u evaluate/benchmark.py > evaluate/results/benchmark.log 2>&1

# follow it from another terminal (Ctrl-C stops the watching, not the run)
tail -f evaluate/results/benchmark.log | grep --line-buffered -E "^===|receipts \(|done in|FAILED|all done"

python evaluate/benchmark.py --compare-only      # rebuild the table from saved runs
```

**Metrics** (`evaluate/metrics.py`)

- **Exact match**, per field and micro-averaged — verbatim string equality.
- **Normalised match** — text case-folded, whitespace-collapsed, surrounding punctuation trimmed;
  totals compared as amounts to the cent (`RM`, `MYR` and `$` dropped, thousands separators
  dropped, a decimal comma read as a point: `RM20,80` = `20.80`); dates compared as calendar dates,
  read day-first as Malaysian receipts print them (`21-05-17` = `21 May 2017`). This is where the
  Phase 1 verbatim decision gets paid off: the cosmetic gap between `RM 11.40` and `11.40` is a
  scoring concern, not a data concern.
- **Field-level precision / recall / F1** on non-null values, so hallucinating a value into a field
  that should be `null` is penalised distinctly from missing one.
- **JSON validity rate** — raw, before repair.
- **Schema adherence** — did the output carry exactly the requested keys, in the requested order?
  Reported separately from value accuracy, because it is the claim the project is actually making.
- **Out-of-schema key rate** — keys the model volunteered that nobody asked for.
- **95% intervals** on the headline scores, by bootstrap over receipts (1,000 resamples).

**Baselines, all on the same 347-receipt test split.** Rows 0, 1 and 2 share the MLX 4-bit base,
so the adapter is the only difference between 0 and 2.

| # | System | Question it answers |
|:--|:--|:--|
| 0 | Phi-3.5-mini-instruct, zero-shot, 4-bit | What does the base model already do, on the weights the adapter runs on? |
| 0b | Phi-3.5-mini-instruct, zero-shot, bf16 | How much of the gap is quantization damage rather than missing skill? |
| 1 | Phi-3.5-mini-instruct, 3-shot, 4-bit | How much of the gain is just formatting demonstration? |
| 2 | **ParseTune adapter** | What did fine-tuning actually buy? |

Baseline 1 exists to keep the headline honest — a fine-tune that only matches few-shot prompting
has taught the model formatting, not extraction. Baseline 0b exists for the same reason, found
while building Phase 3: the 4-bit base garbles years — `15-06-2018` comes out as `15-06-2ized`,
often followed by unrelated prose — in 21 of 62 validation receipts, while the bf16 model copies
the same dates exactly. The adapter, trained on the 4-bit weights, has learned around it. Without
0b, part of the adapter's gain would really be undoing quantization. Every run writes a timestamped JSON into
`evaluate/results/` with per-example predictions kept, so failures are inspectable rather than
just counted. The three-shot examples are three canonical training receipts no longer than the
median, drawn with seed 0 (`sroie-train-00814-v0`, `-00886-v0`, `-00096-v0`).

**Results — test split, 347 receipts, batch 2** (`evaluate/results/comparison_test.md`). Scores in
%; field columns and F1 use normalised match.

| # | System | Exact | Normalised (95% CI) | F1 | company | date | address | total | Valid JSON (raw) | Schema adherence |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | **ParseTune (v2 adapter)** | **93.7** | **94.2 (92.9–95.5)** | **94.4** | 91.9 | 99.1 | 89.3 | 96.5 | 99.7 | 99.7 |
| 0b | Phi-3.5-mini, zero-shot, bf16 | 55.0 | 68.0 (66.0–70.2) | 68.0 | 58.2 | 95.4 | 35.7 | 82.7 | 100.0 | 100.0 |
| 1 | Phi-3.5-mini, 3-shot, 4-bit | 41.4 | 54.3 (51.5–56.8) | 55.2 | 50.1 | 79.8 | 26.2 | 60.8 | 96.5 | 96.0 |
| 0 | Phi-3.5-mini, zero-shot, 4-bit | 33.6 | 44.6 (41.7–47.6) | 47.4 | 39.5 | 57.3 | 20.2 | 61.4 | 88.2 | 87.6 |

No system volunteered out-of-schema keys except the bare 4-bit base (0.6% of outputs).

**What the numbers say**

- **Fine-tuning taught extraction, not formatting.** On the same 4-bit weights, the adapter takes
  normalised accuracy from 44.6 to 94.2; three worked examples only reach 54.3. The intervals are
  far apart.
- **About half of that gap is quantization.** The 4-bit base garbles years (`2ized`) in 120 of 347
  test outputs — 26 of them wander until the token budget runs out — while the bf16 base does it in
  none and reaches 68.0. Against the full-precision base model, ParseTune's gain is 26 points:
  that is the honest headline.
- **The schema is followed.** 99.7% of ParseTune's outputs are valid JSON as generated, with
  exactly the requested keys in order. The single failure copied a quote character from the OCR
  text into an address without escaping it, which no repair can undo.
- **Verbatim copying works.** ParseTune's exact and normalised scores differ by half a point; the
  baselines gain 8–13 points from normalisation, because they reformat amounts and dates.

**Where ParseTune still misses** — 80 of 1,388 fields: address 37, company 28, total 12, date 3.

- 44 are span-boundary disagreements, where the prediction is part of the gold value or the other
  way round. Many are noise in the gold (addresses that begin with a registration number such as
  `(989625-A) NO 18…`, a company cut off mid-number) or genuinely ambiguous boundaries — the brand
  against the legal entity, as in `TONY ROMA'S AEON TEBRAU CITY GRAND COMPANIONS SON BHD`.
- 32 are different values: totals read off the wrong line (a subtotal, or the amount before
  rounding — `RM42.88` against `RM42.90`, where the receipt prints both and the gold is arguable),
  addresses whose fragments come out in OCR line order rather than the gold's reading order, and a
  few store names garbled by OCR.
- 4 are the fields of the one invalid JSON output.

**Batch size and reproducibility.** Every system ran at batch 2, seed 0, both recorded in its
results file, and recomputing the metrics from the saved predictions reproduces every number. A
first attempt at batch 4 ran out of GPU memory on the three-shot prompts (~3,000 tokens each, and
continuous batching holds up to twice the batch while the next prompts load). Its two completed
runs are kept beside the final ones, with `benchmark-batch4-oom.log`, and they measure batch
sensitivity directly: ParseTune changed 3 of 347 outputs between batch 4 and batch 2 (94.4 → 94.2),
the bare base 80 of 347 (44.1 → 44.6). The run also survived the Mac going to sleep mid-benchmark —
generation pauses rather than fails — which is why the command above starts under `caffeinate`.

---

## Phase 5 — Generalization: the schema-instructability test `[DONE]`

**Deliverable:** `evaluate/schema_probe.py --conditions phase5` + the Phase 5 section of
`evaluate/results/REPORT.md`

```bash
caffeinate -i python -u evaluate/schema_probe.py --conditions phase5 --name parsetune-v2 \
    --adapter train/adapters/phi35-sroie-v2                       # ~24 min; one run per system
```

Phase 4 proves the model extracts receipts. This phase probes whether it learned the *instruction*
or just memorised four field names. Four held-out schema conditions, none seen in training
(`prepare_data.py` refuses the names and their concepts):

1. **Renamed fields** — `merchant_name`, `purchase_date`, `amount_due`, each with its SROIE field's
   own description. Same semantics, new keys; the description string is the only bridge.
2. **Reordered and subset** — a random subset of the four in a random (never canonical) order; the
   sanity control, since train augmentation covers it.
3. **Novel fields** — `payment_method`, `gst_id`, `line_item_count`: printed, never supervised.
   `gst_id` and `payment_method` are checked against strict-pattern silver labels on the receipts
   that make them unambiguous (304 and 296); all three are judged in a manual spot-check of 50
   receipts — **done by the AI assistant reading each receipt, not by a human**, every verdict saved
   in `evaluate/results/phase5_spotcheck.json`.
4. **Absent fields** — the four plus `shipping_tracking_number`, whose answer is `null` (the 2 test
   receipts that mention tracking or couriers are excluded).

**Results — test split, 347 receipts, batch 2** (normalised match; precision / recall for the
novel fields against the silver labels):

| Condition | ParseTune v2 | ParseTune v1 | Base, bf16 | Base, 4-bit |
|:--|--:|--:|--:|--:|
| 1. Renamed fields right | **96.1** | 96.1 | 77.8 | 53.3 |
| 2. Subset / order right | **94.1** | 94.3 | 65.9 | 46.6 |
| 2. … requested schema kept | **100.0** | 100.0 | 81.8 | 71.2 |
| 4. Absent field → `null` | **100.0** | 0.0 | 96.5 | 92.8 |
| 4. … four fields still right | 94.3 | 94.5 | 67.5 | 45.4 |
| 3. `gst_id` precision / recall | 100.0 / 89.8 | 75.0 / 75.0 | 99.0 / 99.0 | **98.7 / 98.7** |
| 3. `payment_method` precision / recall | 83.7 / 74.3 | 82.1 / 82.1 | 99.0 / 99.0 | **99.0 / 98.6** |

Manual spot-check of v2 (50 receipts): `gst_id` right on 42 and never a wrong value;
`payment_method` right on 33; `line_item_count` right on 33 (29 strictly).

**What it says — plainly:**

- **The instruction generalises across names and shapes.** Under new keys, in new orders and
  subsets, v2 returns the trained fields almost as well as in their own schema (96 / 94 against
  94 in Phase 4) and keeps exactly the requested keys every time. The base model, given the same
  descriptions, manages 78 / 66 at full precision and volunteers keys nobody asked for — asked
  only for `total`, it added `subTotal`, `tax`, `change`, `itemCount` and `qtyCount`.
- **It does not generalise to new concepts — fine-tuning narrowed the model there.** On fields it
  was never taught, the plain base beats both adapters, and the 4-bit base does as well as the
  full-precision one, so this is not quantisation: it is the fine-tune. The adapter copies text
  where the base answers the concept — `"Cashier"` instead of `"Cash"` for the payment method
  (on the 9 VISA receipts v1 was right on all 9, v2 on 4). Teaching more fields, or mixing general
  instruction data into training, is the obvious next step; until then, a novel field is safer
  asked of the base model.
- **Saying `null` was the base model's to lose.** Told to use `null` for missing fields, the base
  already does so 93–97% of the time. The four-field fine-tune (v1) erased that — 0 of 345 — and
  the absent-field data restored it in v2, to 345 of 345, at a small cost: 4 printed values
  (of 1,380) answered `null` when an absent field was in the request.
- **The absent-field data made v2 more careful, not just more null.** v1 returned 76 wrong GST
  ids; v2 returns none, answering `null` instead on 31 it could not place. On `payment_method` the
  same caution cost recall (82 → 74) — a regression kept and reported rather than tuned away on
  test (`decisions.md`, D-38).

**Early evidence, and the fix.** During Phase 3, one validation receipt was asked for
`amount_due`, `merchant_name`, `purchase_date` and `shipping_tracking_number`. The v1 adapter got
the three renamed fields right, in the requested order — and answered the absent one with
`"BTB09201805190274"`, a document number off the receipt, instead of `null`. The cause was the
data: 1 of v1's 1,114 training examples had a `null` answer. Phase 1 now adds real absent-field
examples (see Phase 1), and the v2 adapter trained on them answers the same request with
`"shipping_tracking_number": null`, the three renamed fields still right. (That check was run after
v2 was chosen; the choice itself used only the probes below.)

**Validation probes** (`evaluate/schema_probe.py`, 62 validation receipts, batch 8). These choose
between adapters; they use field names that neither training nor Phase 5 uses:

| Probe | v1 | v2 (adopted) |
|:--|--:|--:|
| Four SROIE fields exact | 234/248 | 234/248 |
| Renamed fields (`shop_name`, `transaction_date`, `grand_total`) exact | 177/186 | 176/186 |
| Printed fields wrongly answered `null` | 0 | 0 |
| Never-trained absent fields (`passport_number`, …) answered `null` | 0/62 | 62/62 |
| Grounding fields on unseen receipts: values right | 19/82 | 82/82 |
| Grounding fields on unseen receipts: absent → `null` | 0/131 | 131/131 |

The adoption bar — four fields within noise (≥ 231/248), no printed field answered `null`, and at
least 80% `null` on never-trained absent fields — was set before retraining; v2 clears all three.
The grounding rows are labelled by the same patterns as training, so they show the patterns were
learned, not that they are right.

**What this changed for Phase 5.** Training includes fields beyond SROIE's four — the grounding
fields with real values and nulls, and the never-present fields with nulls — so condition 3
measures transfer to new fields rather than a first meeting with any, and condition 4 measures
whether "absent" generalises to a concept (tracking and shipping) kept out of training on purpose.
Phase 5's own names and concepts are refused by an assertion in Phase 1, so none of them was seen.

---

## Phase 6 — Packaging `[DONE — publishing left to the owner]`

**Deliverable:** `notebooks/parsetune.ipynb`, `MODEL_CARD.md`, `evaluate/report.py` →
`evaluate/results/REPORT.md`, `decisions.md`, `memory.md`, and the results at the top of this README.

- **Walkthrough notebook** (`notebooks/parsetune.ipynb`, saved with its outputs): the data, both
  training runs' loss curves, one live extraction with renamed fields and an absent field, and the
  Phase 4 and 5 results loaded from the saved runs. It runs on Apple Silicon — the original plan was
  a Kaggle T4/P100 session, but MLX training does not run on CUDA. The heavy steps (data build,
  training, full benchmarks) appear as commands rather than re-running for hours.
  `pip install -r notebooks/requirements.txt` adds what it needs.
- **Results report** (`python evaluate/report.py`): every Phase 4 and Phase 5 number — micro and
  per-field precision / recall / F1 with intervals, batch-size sensitivity, the selection probes,
  the manual spot-check — rebuilt from the result files, so no number is copied by hand.
- **Adapter card** (`MODEL_CARD.md`): usage, data, procedure, results, limitations, licence notes,
  and the exact upload commands. **The adapter was not published:** that is a public action on the
  owner's Hugging Face account. A PEFT-format conversion for `transformers` users was deliberately
  not shipped — it would need validating against the MLX adapter first (`decisions.md`, D-35);
  `mlx_lm.fuse --dequantize` can export fused weights instead.
- **Decision log and project memory** (`decisions.md`, `memory.md`): every decision with its
  reasons — including the ones that changed and why — plus the phases, architecture and scope.

---

## Setup

```bash
uv venv --python 3.12 .venv               # mlx-lm needs Python 3.11+; macOS ships 3.9
source .venv/bin/activate
uv pip install -r data/requirements.txt   # Phase 1 - data preparation (runs anywhere)
uv pip install -r train/requirements.txt  # Phase 2 onward - MLX, Apple Silicon only
uv pip install -r notebooks/requirements.txt  # optional: the walkthrough notebook
```

(`python3.12 -m venv .venv` with plain `pip install` works the same.)

```bash
python tests/test_pipeline.py   # 9 regression tests, CPU only, ~30 s
```

They pin what every number depends on: the normalisers and metrics, the JSON repair, the
token budget against every gold answer, every stored prompt rebuilding byte for byte, no receipt
in two splits, absent-field examples built only from real text, and Phase 5's names never trained.

Core stack: `mlx` and `mlx-lm` for training and inference, `datasets` for Phase 1.
`train/requirements.txt` pins mlx-lm exactly, because `sft_train.py` drives its trainer from
Python and those internals move between releases.

**Hardware:** an Apple Silicon Mac — MLX runs on the Apple GPU out of unified memory. Built on a
MacBook Pro with an M5 Pro and 24GB: training peaks at 5.6 GB, batch-2 inference at 6–11 GB
(bf16 baseline highest). Run one GPU job at a time — training plus inference together exhausted GPU
memory (the first full training attempt died at iteration ~90 that way) — and start long runs
under `caffeinate`, or the Mac's idle sleep pauses them. The first run downloads Phi-3.5-mini
(~7.6GB) and keeps a 2.0GB 4-bit copy in `models/`.

---

## Dataset and licence

SROIE — *Scanned Receipts OCR and Information Extraction*, ICDAR 2019 Robust Reading Competition.
626 training / 347 test receipts, English, Malaysian retail. Used here for research; check the
competition terms before redistributing any of it.
