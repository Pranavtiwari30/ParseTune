# ParseTune

**Schema-instructable document extraction.** A fine-tune of Phi-3.5-mini (3.8B) that reads a
document and returns *exactly the JSON schema you hand it at inference time* — the fields you
asked for, in the order you asked for them, and nothing else.

Most document-extraction models learn one fixed output shape. ParseTune learns the *instruction*:
the schema travels in the prompt, so a model trained on receipts responds to field names and
orderings it never saw during training. Trained with QLoRA + PEFT + TRL's `SFTTrainer` on the
SROIE receipt corpus, on a single consumer/Kaggle GPU.

```
Input   ->  Extract the following fields from this receipt:
            {"company": "name of the merchant...", "total": "final total amount paid"}
            Receipt text: SANYU STATIONERY SHOP / NO. 31G&33G, JALAN SETIA INDAH ...

Output  ->  {"company": "SANYU STATIONERY SHOP", "total": "4.5000"}
```

---

## Status at a glance

| Phase | What it delivers | State |
|:--|:--|:--|
| **1 — Data** | SROIE to instruction pairs, split, saved as JSON | **Done & verified** |
| **2 — Training** | QLoRA + PEFT SFT of Phi-3.5-mini | Next |
| **3 — Inference** | Constrained generation + JSON repair | Planned |
| **4 — Evaluation** | Metrics, baselines, benchmark harness | Planned |
| **5 — Generalization** | The schema-instructability test | Planned |
| **6 — Packaging** | Kaggle notebook, adapter release, write-up | Planned |

---

## Repository layout

```
ParseTune/
├── data/
│   ├── raw/                     # optional: official SROIE files, for offline runs
│   ├── processed/               # [done] train/validation/test.json + dataset_info.json
│   └── prepare_data.py          # [done] Phase 1
├── train/
│   ├── sft_train.py             # Phase 2
│   ├── config.yaml              # Phase 2
│   └── requirements.txt         # Phase 2
├── evaluate/
│   ├── benchmark.py             # Phase 4 — runs a model over a split
│   ├── metrics.py               # Phase 4 — scoring
│   └── results/                 # Phase 4 — per-run JSON + comparison table
├── notebooks/
│   └── parsetune_kaggle.ipynb   # Phase 6 — end-to-end on a Kaggle T4/P100
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
   straddle two splits.

**Design decisions worth knowing**

- **Schema carries descriptions, not just names.** The prompt gets
  `{"company": "name of the merchant or business issuing the receipt", ...}`. That description is
  the handle the model uses to interpret an unfamiliar field name at inference time. It is the
  whole mechanism behind Phase 5.
- **Schema augmentation on train only.** Each training receipt yields the canonical 4-field example
  plus extra variants with random field subsets in random order (63 distinct schemas across the
  train split). Validation and test stay canonical so scores are comparable run to run.
- **Messages, not pre-rendered strings.** `prompt`/`completion` are chat-message lists.
  `SFTTrainer` applies the Phi-3.5 chat template itself and masks the prompt out of the loss;
  hand-formatting special tokens here would double them.
- **Values are verbatim OCR spans.** `"total": "4.5000"`, not `4.50`. Checked against the official
  task-3 gold on 100 receipts, verbatim spans agree exactly on date 97%, total 94%, company 83%,
  address 60% — and *normalising lowered that*, because the gold is itself inconsistent. Verbatim
  spans are also the only targets actually copyable from the input, so they are what a text-only
  model can learn. Cosmetic differences get folded into the comparison in Phase 4, where they belong.

**Output**

| Split | Receipts | Examples | Schemas |
|:--|--:|--:|--:|
| train | 564 | 1,128 | 63 distinct |
| validation | 62 | 62 | canonical only |
| test | 347 | 347 | canonical only |

Official SROIE test split kept untouched; validation carved from train at 10%.

**Verified:** every `completion` parses as JSON and matches `target`, with key order equal to
`schema` (0 inconsistencies over 1,537 examples); 0 duplicate ids; 0 empty receipt texts; 0
`receipt_id` overlap between any two splits; 1 receipt missing a field corpus-wide.

```bash
python data/prepare_data.py                      # default: Hub -> data/processed/
python data/prepare_data.py --limit 20           # smoke test
python data/prepare_data.py --schema-variants 1  # canonical schema only
python data/prepare_data.py --source local       # offline, from data/raw/
```

`data/processed/dataset_info.json` records the exact template, field specs, counts, and full
argument set used for the run sitting next to it.

---

## Phase 2 — Training `[NEXT]`

**Deliverable:** `train/sft_train.py`, `train/config.yaml`, `train/requirements.txt`

QLoRA fine-tune of `microsoft/Phi-3.5-mini-instruct` — the base weights stay frozen in 4-bit, and
only a small adapter trains, which is what fits 3.8B onto one 16GB GPU.

**Planned configuration** (lands in `config.yaml`, overridable from the CLI)

| Knob | Value | Why |
|:--|:--|:--|
| Quantisation | 4-bit NF4, double-quant, bf16 compute | The QLoRA recipe; ~2.3GB of frozen base weights |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | Standard for a task this narrow |
| Target modules | `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` | Phi-3 uses *fused* projections — the usual `q_proj`/`k_proj` names silently match nothing |
| Max sequence length | 2048 | Covers the receipt body plus schema; the long tail is truncated at 4000 chars upstream |
| Batch / grad-accum | 1 x 8 | Effective batch 8 inside 16GB |
| LR / schedule / epochs | 2e-4, cosine, 3% warmup, 3 | Roughly 420 optimizer steps |
| Optimizer | `paged_adamw_8bit` | Survives the memory spikes |
| Gradient checkpointing | on | Trades ~20% speed for the headroom that makes 2048 tokens fit |

**Loss masking is the thing to get right.** With prompt-completion data, `SFTTrainer` trains only
on the assistant turn. If the loss leaks into the prompt, the model spends its capacity learning to
reproduce receipts instead of extracting from them. Phase 2 includes an explicit check that decodes
one batch and confirms the unmasked span is the JSON answer and nothing else.

**Exit criteria:** training loss decreases and validation loss does not diverge; a smoke run on
`--limit 20` completes end-to-end; the adapter saves and reloads; one greedy generation on a
validation receipt produces parseable JSON.

---

## Phase 3 — Inference `[PLANNED]`

**Deliverable:** `evaluate/predict.py` (loader + generation path shared by Phase 4)

Loads base + adapter, formats a receipt and an arbitrary schema through the same template Phase 1
wrote, and generates.

- Greedy decoding (`do_sample=False`) — extraction has one right answer, sampling only invents.
- `max_new_tokens` sized to the schema, plus a stop condition on the closing brace.
- A **JSON repair pass**: strip markdown fences, cut trailing prose, balance braces. Recorded as a
  *repaired* flag rather than hidden, so Phase 4 can report raw JSON validity honestly.
- Batched generation with left padding, so a 347-receipt benchmark takes minutes, not an hour.

---

## Phase 4 — Evaluation `[PLANNED]`

**Deliverable:** `evaluate/metrics.py`, `evaluate/benchmark.py`, `evaluate/results/*.json`

**Metrics**

- **Exact match**, per field and micro-averaged — verbatim string equality.
- **Normalised match** — case-folded, whitespace-collapsed, currency symbols and thousands
  separators dropped, dates unified. This is where the Phase 1 verbatim decision gets paid off: the
  cosmetic gap between `RM 11.40` and `11.40` is a scoring concern, not a data concern.
- **Field-level precision / recall / F1**, so hallucinating a value into a field that should be
  `null` is penalised distinctly from missing one.
- **JSON validity rate** — raw, before repair.
- **Schema adherence** — did the output carry exactly the requested keys, in the requested order?
  Reported separately from value accuracy, because it is the claim the project is actually making.
- **Out-of-schema key rate** — keys the model volunteered that nobody asked for.

**Baselines, all on the same 347-receipt test split**

| # | System | Question it answers |
|:--|:--|:--|
| 0 | Phi-3.5-mini-instruct, zero-shot | What does the base model already do? |
| 1 | Phi-3.5-mini-instruct, 3-shot | How much of the gain is just formatting demonstration? |
| 2 | **ParseTune adapter** | What did fine-tuning actually buy? |

Baseline 1 exists to keep the headline honest — a fine-tune that only matches few-shot prompting
has taught the model formatting, not extraction. Every run writes a timestamped JSON into
`evaluate/results/` with per-example predictions kept, so failures are inspectable rather than
just counted.

---

## Phase 5 — Generalization: the schema-instructability test `[PLANNED]`

**Deliverable:** `evaluate/schema_probe.py` + a section in the results table

Phase 4 proves the model extracts receipts. This phase probes whether it learned the *instruction*
or just memorised four field names. Four held-out schema conditions, none seen in training:

1. **Renamed fields** — `merchant_name`, `purchase_date`, `amount_due`. Same semantics, new keys.
   The description string is the only bridge.
2. **Reordered and subset** — already covered by train augmentation, so this is the sanity control
   that should score highest.
3. **Novel fields** — `payment_method`, `gst_id`, `line_item_count`. Present in receipt text, never
   supervised. Scored by manual spot-check on a 50-receipt sample.
4. **Absent fields** — `shipping_tracking_number`. The correct answer is `null`. This is the
   hallucination test, and the one most likely to fail.

A model that scores well on 1 and 3 but collapses on 4 is confabulating confidently — worth
reporting plainly rather than burying.

---

## Phase 6 — Packaging `[PLANNED]`

**Deliverable:** `notebooks/parsetune_kaggle.ipynb`, adapter card, results write-up

- One notebook that runs data prep -> training -> evaluation inside a Kaggle T4 x2 or P100 session,
  checkpointing to `/kaggle/working` so a session timeout does not cost the whole run.
- Adapter pushed to the Hub with a model card: training config, metrics table, known failure modes.
- Final README results section with the baseline comparison and the Phase 5 findings.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r train/requirements.txt
```

Core stack: `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets`, `torch`.
Exact pins land with Phase 2 — `bitsandbytes` and `trl` are both version-sensitive, and TRL's
`SFTTrainer` API has changed shape across recent releases.

**Hardware:** one GPU with 16GB VRAM or more (T4, P100, RTX 4080, A10). CPU-only training is not
practical; Phase 3 inference runs on CPU but slowly.

---

## Dataset and licence

SROIE — *Scanned Receipts OCR and Information Extraction*, ICDAR 2019 Robust Reading Competition.
626 training / 347 test receipts, English, Malaysian retail. Used here for research; check the
competition terms before redistributing any of it.
