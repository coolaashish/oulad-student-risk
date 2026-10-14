# Project plan (10 Oct 2026 to Feb 2027)

Goal: a finished, well-explained project plus job applications going out every week.
Assumes about 8 to 10 hours a week alongside the course and carer job. Adjust with your supervisor.

## Phase 1: 10 Oct to 31 Oct, data and baselines
- Create the GitHub repo and push this scaffold (no data files).
- Download OULAD, run `02_baselines.py`, and read every line of `oulad_features.py`.
- Explore the data: class balance, missing values, how results change with `--cutoff` 30 / 60 / 90.
- Learn as you go: pandas and scikit-learn (Kaggle Learn, free).
- CV: send details and I will build it. Start applying to graduate schemes (5 to 10 a week).

## Phase 2: November, LLM cold-start prediction
- Learn Hugging Face Transformers basics and prompt design.
- Render student records as text, test an API model zero-shot and few-shot on students with little history (cold-start subset).
- Compare against XGBoost on the same subset: accuracy, F1, ROC-AUC, latency, cost.
- Record exact model name and version, temperature 0.
- Ask your supervisor about ethics approval for RQ3 now.

## Phase 3: December, RAG feedback and faithfulness
- Build the retrieval step: pull the specific assessments, dates and trends behind each prediction.
- Generate feedback that must cite those records.
- Build the automated faithfulness check: parse each claim and verify it against the retrieved records.
- Start interview preparation (Python, SQL, ML basics, explaining your project).

## Phase 4: January, analysis and write-up
- Error analysis: where does it fail, and for which students?
- Fairness check across groups (gender, age band, deprivation band, disability).
- Add the small open model (Qwen3 8B or Llama 8B) for latency and cost comparison, if time allows.
- Write the dissertation as you go, in your own words.

## Phase 5: February, finish and apply
- Final experiments, clean the repo, polish README, optional demo app.
- Submit. Update CV with real results.
- If no sponsored job yet, apply for the Graduate visa before your Student visa expires and keep applying.

## Always
- Two finished experiments beat five half-done ones.
- Be able to explain every line of code and every method choice.
- Check university rules on AI assistance for the dissertation.
