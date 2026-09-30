# evals/FAILURES.md — failure analysis (REAL observed failures only)

> NOTE: Eval questions in `questions.jsonl` are a first draft and **need human review**
> for factual wording and stability over time before citing results externally.

## Template (fill each section only with failures actually observed in runs)

### 1. Hallucinated / unsupported citation
- Observed in offline mock runs (2026-09-30, `--offline`, n=3×3): mock generator emits placeholder
  reports with zero supported citations (citation_acc 0.000). Expected harness artifact, not a model
  finding — offline mode has no retrieval. No live-model hallucinations observed (no live runs yet).

### 2. Looping / non-terminating critic
- Not observed. Unit test `test_pipeline_revision_cap` forces `revise` × 3 and asserts the loop
  stops at ≤2 revisions with `abort_reason=revision_cap_reached(2)`.

### 3. Poor source selection / weak sources
- Not observed with live retrieval (no live runs yet). Critic `weak_source` issue type exists but untriggered.

### 4. Rate-limit / transient API breakage
- Not observed live (no keys configured). Retry/backoff path covered by unit tests:
  `test_with_retries_eventual_success` (429 → success) and `test_with_retries_non_retryable` (400 → raise).

### 5. Budget aborts / partial results
- Exercised in unit test `test_budget_abort_baseline` (tiny budgets → non-empty `abort_reason` with a
  partial `Report`). No unexpected aborts in offline evals (mock usage stays within defaults).

### 6. Judge disagreement (LLM judge vs human spot-check)
- No live judge runs yet. Spot-check sample generated at
  `evals/results/spotcheck_20260930_212132.csv` (10 rows) — awaiting human verification.
  Judge–human agreement rate: TBD.

## Run log
| date | config | notes |
|---|---|---|
| 2026-09-30 | offline mock × 3 configs | all metrics 0.000 (placeholder reports); harness plumbing verified |
| TBD | live 25q × 3 configs | needs GROQ_API_KEY + judge key |
