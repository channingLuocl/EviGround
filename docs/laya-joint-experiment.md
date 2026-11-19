# Isolated Laya + Qwen experiment

**Follow-up:** the v4-only conclusion below is historical. The complete directed v5-v10 validation,
real image request and three fixed 108-request semantic batteries are reported in
[laya-joint-directed-validation.md](laya-joint-directed-validation.md). The tested division of roles
has a reproducible semantic feasibility obstacle and remains unaccepted; this is not a claim that
all possible Laya interfaces fail. `acceptance.json` is regenerated from all ten frozen cases.

Date: 2026-11-19. Branch: `codex/laya-dual-process`. Baseline: `65ecfe092f1633d553bac4c3de85cf439620eb3c`.

Worktree: `/Users/luocl/.codex/worktrees/laya-dual-process/EviGround`.
Artifacts: `outputs/laya-joint-20261119/` in this worktree only. Never use these as formal or pure-Qwen results.

## Roles and visibility

- Token Plan `qwen3.8-max` performs the original fresh query parse, slow evidence interpretation, explicit semantic Monitor updates and two or three complete next-action proposals.
- Laya English chooses a concrete action among those proposals. It cannot read pixels. Every binding, exclusion, Monitor update, action parameter and eventual prediction comes from an explicit model response. Code only validates and executes the selected action unchanged.
- The current implementation retains the complete original planner state, including uncertainty, accepted DAG, bindings, exclusions and their scope, evidence directory, previously delivered exact text, reports, budgets and feedback. It does not silently summarize or discard this state. The fast contract adds the complete shared Monitor proposal, consultant analysis and complete candidate actions. This is a bounded baseline, not a claim that an arbitrarily long history fits Laya.
- The program derives short option labels from the verb, the model-written reason and literal parameters. Complete parameters remain in state. If distinct full actions cannot be given distinct <=48-token labels, or state/head exceeds the model budget, selection is rejected before HTTP. Neither truncation nor an automatic single-option choice is allowed.
- Qwen restricts the candidate set and proposes the shared Monitor transition; Laya's independent value beyond that restriction has not been demonstrated. Semantic probe failures must not be hidden by saying only JSON/transport checks passed.
- No inherited completed nodes, model responses or predictions. Old parsed fixtures are ignored by the live parser bridge. Evaluation targets are not supplied to either model.

## Source freeze

Read-only origin: `/Users/luocl/Desktop/EviGround/outputs/side_context_mainflow_20261119_01/versions/v28/`.
`frozen/provenance.json` records each source, destination and SHA-256. Sixty selected files totaling 851,999 bytes were copied only after all expected source hashes matched. All Python imports come from the isolated snapshot. The original raw scene clouds and predicted category tables were also hash-checked; results are in `raw-input-checks.json`.

The private Token Plan config is read at runtime; its API key is never copied to code, requests, documentation or logs. SSH authentication likewise remains in memory. The independent venv is `outputs/laya-joint-20261119/.venv`; main venvs and ports are untouched. The existing remote Laya process is reused through a loopback-only SSH tunnel: local `127.0.0.1:18180` to remote `127.0.0.1:18080`, tunnel ID `laya-isolated-34153-18180`.

## Budget mapping

The ceilings remain B=10, R=6, T=17 and 4000 reserved output tokens per primary decision across roles. This remains an exploratory protocol extension, not a controlled comparison with pure Qwen: the six R slots also pay for the added slow semantic consultations.

| Event | Charge |
| --- | --- |
| Final fast action-selection attempt | One T decision, including failures |
| New slow primary semantic consultation | One R, charged before preflight/HTTP |
| Successful physical inspect/measure | One B, original definition |
| Explicit archive read or covered measurement reuse | One R, original definition |
| Invalid action/contract correction | Original additional R charge |
| Visual worker after physical/consult action | Original bundled role, shares that decision's 4000 output reservation |
| Original parser calls | Original separate stage; original attribute repair remains separate |
| Added parser catalog repair | One main R, original frozen bridge rule |
| Laya output reservation | Zero generated tokens, but its decision attempt and model receipt are logged |

Every fast/slow/visual/parser call, failed send and preflight rejection is retained. There are no previous-call caches. An HTTP/provider transport failure now stops for inspection rather than issuing an automatic retry. No new GPU model, instance or local Qwen service was launched.

## Capacity

Laya: default English 512 and multilingual 1024 are configurable budgets. The installed HTTP ceiling is 8192. Actual English input_tokens=7924 was untruncated; an overlength state lost 132 tokens at the 8192 clamp; 8193 was rejected. This is capacity accounting, not semantic accuracy. Full report: `capacity-validation.json`.

Before every Laya send, the client uses the actual checkpoint tokenizer to count complete state, head and each option, enforcing <=8192 and <=48 tokens per option. After the send it verifies exact counts, zero dropped state tokens and truncated=false. The complete options are retained in state.

Qwen: every request preflight includes all messages, the full actual wire response schema, each actual decoded image and reserved output. No Qwen2-VL surrogate tokenizer is used. Text is bounded conservatively by UTF-8 byte count under the explicit byte-level BPE assumption, with 1024 extra framing tokens; each image reserves the official Qwen3.8 ceiling of 16384 visual tokens plus two markers. Image bytes/dimensions/SHA-256 are checked from actual attachments. The method is deliberately conservative, not an exact closed-model tokenizer. Every accepted response must report prompt_tokens below the input bound, and prompt_tokens plus the reserved output must remain <=32768. This method can reject requests that would fit an exact tokenizer; such rejection is a verification barrier, not proof that the provider model lacks capacity.

Official vision rule source: <https://help.aliyun.com/zh/model-studio/vision>, retrieved locally as `qwen-vl.html` and `qwen-vl.txt`. Visual role capacity is not considered empirically accepted until a real image request completes and its usage is verified.

Repeated response-schema fragments are shared through JSON Schema `$defs` without changing allowed answers. The exact factored schema is locally meta-validated before sending. This lossless serialization does not remove facts or enlarge budgets.

## Checks and diagnostics

- Offline replay: nine tests cover wrong types/IDs, stale state versions, target/reference domains, non-distinct actions, no-truncation capacity rejection, actual-image/schema/output accounting, rollback and rival reactivation, schema reference correctness, cross-node alternatives and slow-slot limits. Logs: `offline-replay-v4.log` and `offline-replay.json`.
- Synthetic probes use known artificial evidence, never evaluation labels, and execute no physical tools. They are separate diagnostic calls, not free extra steps in a real episode.
- Probe v1 used opaque labels; all 12 choices selected c0, so it did not demonstrate meaningful action selection. It also confounded evidence position with order and is not an independent position test.
- Probe v2 crosses three evidence positions with three option orders, four cases (36 requests). Short literal action labels still produce unstable choices; strict expected-action matches are 11/36.
- Probe v3 uses natural-language short reasons plus literal arguments, the same 36-case design. Strict expected-action matches are 13/36. These strict counts are not an accuracy benchmark: for missing visual evidence, both inspection and consulting an existing image can be defensible. More decisive failures remain: in the unknown-material case the model repeats geometry in 6/9 cases; in the missing-geometry case it chooses measurement in only 4/9 cases. All responses were untruncated and count-checked. This does not establish reliable fast planning or long-context semantics.
- Real v1 Nr3D: fresh parser generation reported exact `qwen3.8-max`; the first primary request was stopped before HTTP because the conservative all-input bound was 33890>32768. B=0, R=1, T=1. No Laya task action or prediction was emitted.
- Real v2 Nr3D: lossless schema factoring brought the first primary preflight below 32768, but the provider rejected a malformed `$ref` inserted into a `properties` map. The repeated rejection stopped the episode at B=0, R=4, T=2. Both failures are retained. The factorer now only replaces actual schema nodes, validates the full factored contract locally, and stops immediately on provider failure.
- Real v3 Nr3D: actual slow generations now pass the provider schema and report `qwen3.8-max`, but each response includes an object-geometry alternative for node n1 while the shared Monitor's current node is n0. The whole alternative set is rejected; no illegal action is executed. The initial feedback was only `AssertionError`; the precise cross-node error and explicit same-node instruction were then added, offline replayed and refrozen. B=0, R=4, T=2.
- Real v4 Nr3D: the first actual fast/slow action chain executes. Qwen proposes three alternatives and a pending DAG; Laya chooses c2, an explicit initial-source native-row consultation for IDs [12,13,0,1]. The Monitor structurally accepts it and the exact four rows are delivered without a new physical acquisition. The Laya request uses 4352 state tokens +130 head tokens +1 closing token =4483; preflight matches usage exactly, with zero dropped tokens. Option lengths are [36,38,33], all below 48.
- On the next v4 decision, the complete Qwen wire request has a conservative bound of 39246 including the 2200 reserved output; it is rejected before HTTP. This is not B/R/T exhaustion and is not proof that qwen3.8-max's actual model window would overflow. The raw runner status says `run_budget_exhausted`; the posthoc acceptance report normalizes it to `capacity_preflight_blocked`. The last completed slow generation reported only 3475 actual prompt tokens, so the byte bound is explicitly conservative; no exact count for the blocked request is claimed.
- The same Token Plan endpoint's `/tokenize` and `/count_tokens` probes both return HTTP 400 URL errors. This proves those two tested routes are unusable, not that the provider has no possible counting interface. Evidence: `tokenizer-capability-probe.json`.

## Acceptance conclusion

Current status: **blocked, not accepted**. Final source freeze: v4 (including a complete copy of the entry script). All versions remain separate fresh cases; no old prediction or completed node was inherited. The earlier v1-v3 entry hashes and tool edit history are retained, but only v4 includes the full entry-script copy inside its version directory.

| Acceptance surface | Observed result |
| --- | --- |
| Deployment | Usable; original English/multilingual models resident on the existing 4090 |
| Actual model identity | Parser and slow generations report exact `qwen3.8-max`; fast routing reports English checkpoint |
| Real fast/slow link | One Laya-selected archive consultation accepted and executed in v4 |
| Complete real task | Failed to complete; stopped at next request's capacity preflight |
| v4 B/R/T | B=0, R=3, T=2; no ceiling exceeded |
| Independent evidence audit | Insufficient: no actual image observations, completed reference binding, verified relation or final evidence |
| Localization correctness | No prediction emitted; evaluation GT not read; no accuracy score claimed |
| Invalid/repeated scene actions in v4 | Zero invalid or repeated executed actions; one rejected preflight attempt |
| State semantics | All accepted nodes remain pending; no target exclusions or program-completed nodes |
| Fast semantic reliability | Not established; short-contract failures and option-order sensitivity are retained |
| Multimodal request capacity | Unverified in real execution: no actual visual-role model request occurred |
| Second dataset | Not started, because the first dataset did not pass |

`acceptance.json` contains the separate process, evidence, correctness, invalid-action and capacity conclusions with trace hashes. The posthoc audit reads no evaluation labels and invokes no models. The next unresolved work is a defensible tighter provider token counter or a model-selected, source-traceable focused contract, plus reliable short-contract action selection. Dropping constraints, auto-selecting targets/completing nodes, increasing budgets or calling a prior partial trace a success would not resolve these blockers.

## Reproduce

From the isolated worktree, with the existing safe SSH tunnel active:

```bash
outputs/laya-joint-20261119/.venv/bin/python scripts/check_laya_joint.py
outputs/laya-joint-20261119/.venv/bin/python scripts/laya_joint.py prepare --version NEW_VERSION
outputs/laya-joint-20261119/.venv/bin/python scripts/laya_joint.py run --version NEW_VERSION --dataset nr3d
```

Preparation and execution refuse existing version/case directories. The entry script, generated runner, source snapshot and tokenizer must still match their frozen SHA-256 values. Refreeze after any implementation change; never resume an exhausted case or reuse a prior prediction. Do not start ScanRefer until a fresh Nr3D episode passes the separate acceptance checks.
