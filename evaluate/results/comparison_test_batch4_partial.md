# ParseTune — test split (347 receipts, batch 4)

Scores in %. Field columns are normalised match; F1 is normalised.

| System | Exact | Normalised (95% CI) | F1 | company | date | address | total | Valid JSON (raw) | Schema adherence | Out-of-schema keys |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| parsetune | 93.8 | 94.4 (93.1–95.7) | 94.5 | 92.2 | 99.1 | 89.6 | 96.5 | 99.7 | 99.7 | 0.0 |
| zero-shot | 33.0 | 44.1 (41.3–47.1) | 46.9 | 39.2 | 56.2 | 20.2 | 60.8 | 88.2 | 87.6 | 0.6 |
