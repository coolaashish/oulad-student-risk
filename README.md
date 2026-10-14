# Evidence-Grounded Student Risk Prediction (ML + LLM)

MRes AI project. A hybrid system that (1) predicts which students are at risk using a fast classical model, and (2) generates natural-language feedback with an LLM, where every claim is grounded in, and checked against, the student's actual records.

## Research questions

1. **RQ1** Can a hybrid model match the accuracy of classical baselines on real longitudinal student data?
2. **RQ2** Can retrieval-augmented generation (RAG) produce feedback that is measurably grounded in a student's assessment history?
3. **RQ3** How do students and instructors rate evidence-grounded feedback versus a bare risk score? (needs ethics approval; confirm feasibility with your supervisor)
4. **RQ4** What are the latency and cost of the hybrid system compared with a classical-only baseline?

## Data

- **OULAD** (Open University Learning Analytics Dataset): download the zip from the official OULAD page and unzip the CSVs into `data/oulad/`. Check the licence and terms on that page before use, and before sending any records to an external API.
- **UCI Predict Students' Dropout and Academic Success**: secondary validation dataset (added later).

Do not commit the data to GitHub (see `.gitignore`).

## Setup

```bash
pip install -r requirements.txt
```

Works in Google Colab, VS Code, or a local Python 3.10+ environment.

## Run

```bash
python 02_baselines.py --data-dir data/oulad --cutoff 60
```

This builds features using only information available up to day 60 of each course (no outcome leakage), then compares a majority-class dummy, logistic regression and XGBoost with 5-fold cross-validation grouped by student. Results are printed and saved to `baseline_results.csv`.

## Design decisions (be ready to explain these in interviews)

- **Target:** `at_risk` = final result is Fail or Withdrawn.
- **Cut-off day:** a useful early-warning system must predict *before* the outcome is known, so features only use assessments and VLE activity up to the cut-off. Changing `--cutoff` shows how accuracy improves as more evidence arrives.
- **Students who withdrew before the cut-off are excluded**, because their outcome is already known.
- **Grouped cross-validation by student:** the same student can appear in several module presentations, so rows are never split across train and test for the same student.
- **Timing is recorded** for every model, because RQ4 compares latency and cost.

## Project files

| File | Purpose |
|---|---|
| `oulad_features.py` | Loads OULAD tables and builds leakage-safe features at a cut-off day |
| `02_baselines.py` | Baseline models, grouped CV, metrics and timing |
| `PLAN.md` | Schedule from October to February |

## Next steps

LLM cold-start prediction, RAG feedback generator, faithfulness check, error and fairness analysis. See `PLAN.md`.
