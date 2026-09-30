# Judge prompts (version: v1)
## Coverage judge
Given a must_have_fact and a report, output STRICT JSON: {"fact": "...", "present": true|false, "evidence": "<short quote or empty>"}.
Mark present=true only if the report states the fact explicitly or unambiguously entails it.

## Entailment judge
Given a claim, its supporting_quote, and the source page excerpt, output STRICT JSON: {"claim_index": N, "verdict": "supported|partially|unsupported", "reason": "..."}.
supported = excerpt directly states the claim. partially = implies part of it. unsupported = otherwise.
