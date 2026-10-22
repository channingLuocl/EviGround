# ReAct JSON action interface experiment — 2026-10-22

The original full7805 and independent192 runs continue on the unchanged runtime_eb261ce.

Observed issue: several responses claim inspection/measurement tools are unavailable while the recorded evidence budget is open. The application supports JSON actions, rather than native API tool calls.

The change only explains that interface and exposes available_actions consistent with the action schema and remaining budget. Candidate selection, model, controller guards and evidence budget are unchanged.

Five cases were selected for the observed interface misunderstanding, regardless of correctness. Both baseline5 and contract5 are fresh runs using identical rows, predicted Top-5 inputs and GPT relay settings. These are mechanism stress tests, not a representative accuracy estimate and not a replacement for all 7,805 rows. Do not merge either arm with previous correct outcomes.

See protocol.json, stress5_manifest.json and the changed runtime source_manifest.json for provenance. Credentials are read only by the existing launcher from the external profile.

## Relay incident and partial observations

On 2026-10-22, both experimental arms and the original runs encountered shared HTTP 502 responses. At 04:08 UTC the four verified driver PIDs received SIGSTOP to hold new dispatch while existing workers finished. Text, one-image and four-image health probes subsequently returned HTTP 200. At 04:13 UTC the same driver PIDs received SIGCONT. No inference source, cached outcome or failure record was replaced.

`relay_502_dispatch_hold_20261022T040814Z.json`, `relay_health_20261022T040820Z.json`, `relay_image_health_20261022T041118Z.json` and `relay_502_dispatch_resume_20261022T041322Z.json` record that intervention. These are execution health checks, not benchmark predictions.

At the observation in `mechanism_observation_20261022T041533Z.json`, contract5 has all five evaluations: one correct result and four fatal errors. Its five first model actions are inspect. Baseline5 has four recorded evaluations, all fatal errors, and four first actions of finish; the fifth case is still pending at that timestamp. This is evidence about the JSON action interface only. Availability failures confound the accuracy comparison, and all five selected cases remain in each arm's denominator.

The original full7805 and independent192 jobs continue using their unchanged source snapshot. The full 7,805-row single-pass target remains unverified.
