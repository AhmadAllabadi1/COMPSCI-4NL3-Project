# Codabench Starter Kit

## Files

| File | Description |
|------|-------------|
| `train.csv` | Training set (id, text, label) — use this to train your model |
| `validation.csv` | Validation set (id, text, label) — use this to evaluate during development |
| `test.csv` | Test set (id, text) — no labels, used to generate your submission |
| `baseline_majority.py` | Majority class baseline script |
| `baseline_logreg.py` | TF-IDF + Logistic Regression baseline script |

## Labels

Each sample is classified into one of five categories:
- `ADVICE`
- `WARNING`
- `EMOTIONAL_SUPPORT`
- `ANECDOTE`
- `APPRAISAL`

## Running the Baselines

From inside this directory:

```bash
python baseline_majority.py
python baseline_logreg.py
```

Both scripts print validation set metrics to the console and output a `submission.csv` file.

## Submission Format

Your submission should be a CSV with two columns:

```
id,label
123,ADVICE
456,WARNING
...
```
