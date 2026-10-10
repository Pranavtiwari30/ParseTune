# ParseTune

**Schema-instructable document extraction.** A fine-tune of Phi-3.5-mini (3.8B) that reads a
document and returns exactly the JSON schema you hand it at inference time: the fields you asked
for, in the order you asked for them, and nothing else.

Most document-extraction models learn one fixed output shape. ParseTune learns the instruction.
The schema travels in the prompt as `{field: description}`, so a model trained on receipts
responds to field names and orderings it never saw during training. It was trained with QLoRA on
Apple's MLX (`mlx-lm`) on the SROIE receipt corpus, locally on an Apple Silicon MacBook (M5 Pro,
24 GB).

```
Input   ->  Extract the following fields from this receipt:
            {"company": "name of the merchant...", "total": "final total amount paid"}
            Receipt text: SANYU STATIONERY SHOP / NO. 31G&33G, JALAN SETIA INDAH ...

Output  ->  {"company": "SANYU STATIONERY SHOP", "total": "4.5000"}
```

All six phases (data, training, inference, evaluation, generalisation, packaging) are complete.
The adapter weights are not published yet, so running the model means training it first.

## Results

SROIE's official test split: 347 receipts, never used for any choice. Scores in %, normalised
match with 95% intervals.

| System | Normalised | Exact | F1 | Valid JSON |
|:--|--:|--:|--:|--:|
| **ParseTune (v2 adapter)** | **94.2** (92.9–95.5) | **93.7** | **94.4** | **99.7** |
| Phi-3.5-mini, zero-shot, full precision | 68.0 (66.0–70.2) | 55.0 | 68.0 | 100.0 |
| Phi-3.5-mini, 3-shot, 4-bit | 54.3 (51.5–56.8) | 41.4 | 55.2 | 96.5 |
| Phi-3.5-mini, zero-shot, 4-bit | 44.6 (41.7–47.6) | 33.6 | 47.4 | 88.2 |

Fine-tuning adds 26 points over the full-precision base model. The larger gap to the 4-bit base
is partly quantisation damage, not skill.

Schema-following on field names it never trained on (same test split):

| Request | ParseTune | Base model, full precision |
|:--|--:|--:|
| Renamed fields answered right | **96.1** | 77.8 |
| Random subset in random order answered right | **94.1** | 65.9 |
| Requested keys kept, nothing added | **100** | 81.8 |
| Absent field answered `null` | **345/345** | 333/345 |
| Never-taught `gst_id`, recall | 89.8 | **99.0** |
| Never-taught `payment_method`, recall | 74.3 | **99.0** |

**The limit:** on concepts it was never taught, the fine-tune is worse than the base model it
started from. It copies nearby text ("Cashier") where the base answers the concept ("Cash"). The
instruction generalises across names and shapes, not to new concepts.

Every number, with per-field precision, recall and F1, is in
[`evaluate/results/REPORT.md`](evaluate/results/REPORT.md).

## Setup

Needs an Apple Silicon Mac: MLX runs on the Apple GPU. Built on an M5 Pro with 24 GB; training
peaks at 5.6 GB.

```bash
uv venv --python 3.12 .venv                   # mlx-lm needs Python 3.11+; macOS ships 3.9
source .venv/bin/activate
uv pip install -r data/requirements.txt       # data preparation (runs anywhere)
uv pip install -r train/requirements.txt      # training and inference (MLX)
uv pip install -r notebooks/requirements.txt  # optional: the walkthrough notebook

python tests/test_pipeline.py                 # 9 regression tests, CPU only, about 30 s
```

`python3.12 -m venv .venv` with plain `pip install` works the same. The first training run
downloads Phi-3.5-mini (about 7.6 GB) and keeps a 2.0 GB 4-bit copy in `models/`.

## Usage

```bash
python data/prepare_data.py                # build the dataset (a copy is in data/processed/)
caffeinate -i python train/sft_train.py    # train the adapter, about 2 h 15 min

# extract any schema from one receipt
python evaluate/predict.py --adapter train/adapters/phi35-sroie-v2 \
    --receipt receipt.txt --schema '{"merchant_name": "name of the shop"}'
```

Run one GPU job at a time, and start long runs under `caffeinate` so the Mac does not sleep
through them. The benchmark, the generalisation probes, the report and every other command are
in [`memory.md`](memory.md#6-reproduce).

## Repository layout

```
data/         prepare_data.py and the processed dataset (train / validation / test)
train/        sft_train.py, config.yaml, and each adapter's run log (weights are not in git)
evaluate/     predict.py, benchmark.py, metrics.py, schema_probe.py, report.py, results/
notebooks/    parsetune.ipynb, an executed end-to-end walkthrough
tests/        test_pipeline.py, regression tests for metrics, JSON repair, prompts and data
```

## Where to read more

| File | What it holds |
|:--|:--|
| [`memory.md`](memory.md) | The project in detail: architecture, what each phase did and found, where every artifact lives, how to reproduce it, and what is still open |
| [`decisions.md`](decisions.md) | Every decision, D-01 to D-42, with the options, the reasons, and what changed the ones that changed |
| [`evaluate/results/REPORT.md`](evaluate/results/REPORT.md) | All results: per-field scores, batch-size sensitivity, the held-out schema conditions, the spot-check, model selection |
| [`notebooks/parsetune.ipynb`](notebooks/parsetune.ipynb) | A walkthrough with the loss curves and one live extraction |

## Limitations

- Trained and tested on English Malaysian retail receipts from 2017–2018. Other receipt styles,
  languages and document types are untested.
- The input is OCR text, not images.
- Fields unlike company, date, address and total are answered less reliably than by the base
  model (see the limit above).
- Trained on and meant for the MLX 4-bit base model.
- A value copied from text containing a `"` can break the JSON (1 of 347 test receipts).

The full list, with causes and candidate fixes, is in
[`memory.md`](memory.md#7-open-items-and-known-limitations).

## Dataset and licence

SROIE (*Scanned Receipts OCR and Information Extraction*), ICDAR 2019 Robust Reading Competition:
626 training and 347 test receipts, English, Malaysian retail. Used here for research; check the
competition terms before redistributing any of it. The base model, Phi-3.5-mini-instruct, is
MIT-licensed.
