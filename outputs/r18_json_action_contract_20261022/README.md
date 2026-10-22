# ReAct JSON action interface experiment — 2026-10-22

The original full7805 and independent192 runs continue on the unchanged runtime_eb261ce.

Observed issue: several responses claim inspection/measurement tools are unavailable while the recorded evidence budget is open. The application supports JSON actions, rather than native API tool calls.

The change only explains that interface and exposes available_actions consistent with the action schema and remaining budget. Candidate selection, model, controller guards and evidence budget are unchanged.

Five cases were selected for the observed interface misunderstanding, regardless of correctness. Both baseline5 and contract5 are fresh runs using identical rows, predicted Top-5 inputs and GPT relay settings. These are mechanism stress tests, not a representative accuracy estimate and not a replacement for all 7,805 rows. Do not merge either arm with previous correct outcomes.

See protocol.json, stress5_manifest.json and the changed runtime source_manifest.json for provenance. Credentials are read only by the existing launcher from the external profile.
