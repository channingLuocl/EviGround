# GPT-6.1 relay continuation

The existing R18 Qwen experiment remains frozen. Its ReAct controller, predicted
Top-5 categories, inspect/measure tools, and three-tool evidence budget are copied
unchanged into `source_r18_gpt61`. Only its API adapter changes.

- Actual relay model discovery returned `gpt-6.1-sol`.
- The real image/JSON probe passed using JSON-object mode and medium reasoning.
- No JSON Schema is sent in the wire response_format. The schema stays in the
  task instructions and is checked locally with jsonschema.
- A request gets at most three format attempts, with validation feedback.
  A final object selection without evidence about that object also fails validation.
- Neither format repair nor query allocation receives GT categories or target IDs.
- The credential profile is outside Git, at
  `/Users/luocl/.config/seeground/gpt61-relay.env`, with mode 0600.

`validate_gpt61_promotion.py` first runs the complete development96 cohort, then
the disjoint validation192 cohort. The gates remain 68/96 and 127/192. A prefix,
a best-of merge, or execution-error deletion cannot satisfy either gate.

`gpt61_continuation.py --watch` monitors only explicit quota-exhaustion errors.
Rate limiting, timeouts, and schema failures do not trigger a switch. At exhaustion
it stops the Qwen coordinator and the older supervisors, lets existing workers
drain, and evaluates those already-started queries without reissuing inference.
Every Qwen outcome stays, including failures. Only queries never started by Qwen
are allocated to GPT. GPT starts after both promotion gates pass.

The resulting evaluation must be labeled a **quota-triggered mixed-model single
pass**, not a pure-Qwen or pure-GPT score. The original Qwen run remains a separate
partial experiment if the quota switch occurs.

After the GPT continuation finishes, run `audit_gpt61_quota_full.py`. It requires
all 7805 official queries, disjoint model allocations, actual request settings,
unchanged source hashes, false GT-input flags, actual ReAct final choices, and all
execution failures retained. Only an audited score of at least 5152/7805 satisfies
the 66% target. Completion still requires final review and delivery.
