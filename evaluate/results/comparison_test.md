# ParseTune — test split (347 receipts, batch 2)

Scores in %. Field columns are normalised match; F1 is normalised.

| System | Exact | Normalised (95% CI) | F1 | company | date | address | total | Valid JSON (raw) | Schema adherence | Out-of-schema keys |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| parsetune | 93.7 | 94.2 (92.9–95.5) | 94.4 | 91.9 | 99.1 | 89.3 | 96.5 | 99.7 | 99.7 | 0.0 |
| zero-shot | 33.6 | 44.6 (41.7–47.6) | 47.4 | 39.5 | 57.3 | 20.2 | 61.4 | 88.2 | 87.6 | 0.6 |
| three-shot | 41.4 | 54.3 (51.5–56.8) | 55.2 | 50.1 | 79.8 | 26.2 | 60.8 | 96.5 | 96.0 | 0.0 |
| zero-shot-bf16 | 55.0 | 68.0 (66.0–70.2) | 68.0 | 58.2 | 95.4 | 35.7 | 82.7 | 100.0 | 100.0 | 0.0 |
