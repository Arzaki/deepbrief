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
| 2026-10-01 | live Groq, multi-critic via UI | 404 `model_not_found` on default cheap model `llama-3.1-8b-instant` (retired from plan; live `/models` listed 11 IDs, no llama-3.x). Retry logic correctly did NOT retry a 404. Added `scripts/check_models.py` + a friendly UI error for future retirements |
| 2026-10-01 | live Groq, baseline via CLI | 400 `tool_use_failed` on first replacement `openai/gpt-oss-120b`: it emits NATIVE `container.exec` tool calls instead of plain text, incompatible with our text `TOOL:` protocol. Switched Groq defaults to `qwen/qwen3.8-27b` (follows the text protocol; full pipeline verified live: critic pass, 0 revisions) |
| 2026-10-01 | live Groq, qwen via CLI | 429 "Request too large ... OTPM Limit 1000, Requested 2048": free-tier cap on per-request output tokens. Added `LLM_MAX_TOKENS=1000` (threaded through all agents/judges) + fail-fast (no retry) for size-based 429s |
| 2026-10-01 | live, DDG search path | `duckduckgo-search` v8 returned ZERO results for every query (silent package rot; also spammed a rename RuntimeWarning per call). Switched dependency to successor `ddgs` v9: warnings gone AND real results returned. Tavily key not needed for now |
| 2026-10-01 | live Groq qwen, multi-critic via UI ("biplor vs borderline") | Researcher flailing: 27 searches / 2 fetches, ~20 near-identical searches for user typo "biplor" with empty results, never fetched or finalized → `llm_call_budget_exceeded(30>=30)`, no report. Fixed: hard cap of 3 searches per sub-question + duplicate-query block + empty-result guidance (all deterministic, in code not just prompt), researcher prompt v1→v2, `MAX_LLM_CALLS` 30→40 (a 5-question plan needs ≥33: planner 1 + 5×6 + writer + critic). 3 new unit tests |
| 2026-10-16 | live Groq qwen, quick via UI ("neural networks", same question twice) | Voluntary surrender: 4 researchers searched (results returned) but never fetched once — each FINALed empty citing "no fetched pages". The fetch-forcing gate was neutered (FINAL always lands on the last quick turn; its message is discarded). Deeper trace forensics found the TOOL parser only honored a *leading single* TOOL: chattered tools never ran, and a 2nd TOOL line got glued into the first query as garbage; plus fetches landing on the final turn stranded evidence with no turn left to cite it. Fixed: (1) eager auto-fetch of top result per search (same turn, zero quota); (2) multi-TOOL parsing (up to 3/turn, known tools only); (3) one bonus turn when fresh evidence lands late; prompt v5→v6. Verified live on the identical question: 0 claims → 21 claims, 3 sections, 17 sources, $0.05 |
| 2026-10-19 | public demo (Render free tier, multi-critic quick) | Run stretched to 2281s (~38 min): (a) Groq 429 storms honor Retry-After, so each call can burn minutes in backoff when the key's quota is gone; (b) the research phase waited on futures SEQUENTIALLY (`fut.result(timeout=240)` per sub-question) — the global wall budget only stamped an abort reason and never ended the phase; queued sub-questions each stacked their own 240s wait. Fixed: one `concurrent.futures.wait` bounded by min(remaining wall, N×researcher_timeout/workers) for ALL futures (missing ones → `researcher_deadline` partial), `MAX_WALL_SECONDS=300` on the public demo. 1 new + 1 updated unit test (74 total). Validated earlier same day: demo live at first hit (200) after ~50s cold start |
| 2026-10-01 | live Groq qwen, multi-critic via UI (bipolar retry) | Run died after 240s with raw `JSONDecodeError: Expecting ',' delimiter (char 4812)`: writer/critic parsed model JSON with zero protection, so one malformed reply (near-certainly a verbatim quote with raw `"`/newlines inside a JSON string) killed the whole run. Fixed: `parse_json_lenient` + shared retry-with-error-feedback helper for writer/critic (2 retries, then clear RuntimeError); researcher FINAL uses lenient too; writer/critic prompts v1→v2 with no-raw-quotes rule. 5 new tests |
| 2026-10-01 | live Groq qwen, standard via CLI (LDAC retry) | Run exceeded 600s with only a plan to show. Per-turn trace forensics: every successful LLM call returned in <2.3s, separated by 118-202s silent gaps — the signature of 429 backoff sleeps, not slow inference. Three parallel researchers bursting on a hot free tier (after a night of heavy testing) tripped rate limits serial runs never hit; retries slept invisibly. Fixed: (1) on_retry hook logs every backoff wait as an `llm_retry` trace event; (2) researcher LLM errors → partial continuation with per-SQ notes in abort_reason; (3) writer falls back to a deterministic cited report and critic to pass-with-note instead of crashing the gathered research; (4) researcher abort timeout (240s) + per-turn trace saves so killed runs keep forensics. Mock-test proves a total quota outage now returns a labeled partial, not a crash |
| 2026-10-01 | live Groq qwen, standard multi via UI ("LDAC vs LHDC") | 10-minute run for a surface-level question: researchers ran strictly serially (~25 serial LLM calls on growing transcripts, 0.2s degrading to 14s/call on free tier). Fixed: thread-pool researchers (`RESEARCH_WORKERS=3`, locks added to Budgets/RunTrace/DiskCache/MockProvider), `--depth quick` gear (3 turns/SQ, no critic; UI default), per-request `LLM_TIMEOUT_S=90`. Measured on the same question via CLI quick: 60s wall, 11 LLM calls, $0.02 |
| 2026-10-01 | same LDAC run, report inspection | Citation collision: three different pages all listed/cited as [S1] — each researcher minted its own S1..Sn namespace and they collided on merge. Fixed: pipeline passes one shared thread-safe SID minter to all researchers (`mint_sid`); standalone calls keep local namespaces. New test asserts S1..S4 uniqueness |
| TBD | live 25q × 3 configs | needs judge key for cross-family judging |
