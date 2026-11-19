# Laya deployment and isolated exploration

Updated: 2026-11-19 (Asia/Shanghai).

## Isolation record

- Branch: `codex/laya-dual-process`.
- Worktree: `/Users/luocl/.codex/worktrees/laya-dual-process/EviGround`.
- Committed baseline: `65ecfe092f1633d553bac4c3de85cf439620eb3c`.
- Source checkout: `/Users/luocl/Desktop/EviGround`, branch `codex/reproduce-seeground-scanrefer`.
- Source checkout contains uncommitted Qwen changes. No checkout, stash, reset, commit, or transfer of those changes was performed.
- Frozen uncommitted dependencies: selected v28 prototype interfaces, all explicitly listed `src/` dependencies, and two model-safe input fixtures were copied after verifying the source manifest's SHA-256 values. The 60-file, 851,999-byte snapshot and individual provenance are in `outputs/laya-joint-20261119/frozen/`. Runtime imports this snapshot, never the changing main checkout.
- Deployment is complete. Independent Laya + Token Plan qwen3.8-max integration and bounded real-task diagnostics now exist; see [joint experiment](laya-joint-experiment.md). No joint success is claimed from the deployment examples.

## Remote deployment

- Host: `root@connect.bjb1.seetacloud.com`, SSH port `34153`.
- Hardware verified: NVIDIA GeForce RTX 4090, 24,564 MiB; idle before this deployment.
- Base runtime verified: Ubuntu 22.04.5, Python 3.12.3 at `/root/miniconda3/bin/python`, PyTorch 2.8.0+cu128, CUDA available.
- Deployment directory: `/root/autodl-tmp/laya-deploy`.
- Dedicated virtual environment: `/root/autodl-tmp/laya-deploy/.venv`.
- The virtual environment has system-site-packages access to reuse the existing PyTorch installation; deployment installs remain inside this virtual environment.
- Service entry point: official `laya.serve`, local loopback port `18080` (verified free before installation).
- Package installation completed: `laya[serve]==0.4.37`, transformers 5.23.0, vegaml 0.16.5. An initial transformers<5 constraint was incompatible with vegaml and was removed before successful installation.
- Install log: `/root/autodl-tmp/laya-deploy/install.log`.
- Credentials are not stored in this record or the repository.

## Current status

Latest SSH recheck on 2026-11-19 succeeded: RTX 4090, 24,564 MiB total, 6,497 MiB used at inspection;
both Laya checkpoints on CUDA, zero CPU fallbacks. The deployed service was reused. One actual image
worker request now completes in the isolated joint experiment; the full task is still unaccepted.
Directed results: [laya-joint-directed-validation.md](laya-joint-directed-validation.md).

Deployment verified. Both checkpoint weights passed official SHA-256 checks, both models are resident on CUDA, and the English and Chinese ticket examples returned technical without truncation. Service PID at acceptance: 2515; observed GPU memory: 3,641 MiB. A subsequent SSH health check confirmed both CUDA checkpoints, zero CPU fallbacks and the same memory usage. Actual Token Plan parser generations now report the exact model `qwen3.8-max`; separate joint diagnostics have not established a successful episode.

## Joint experiment scope

- Use the existing private Token Plan configuration without copying its key. Both model listing and actual parser generations confirm the exact ID `qwen3.8-max`. Endpoint: `https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1`.
- First complete one fresh real task end to end; audit evidence sufficiency and prediction correctness separately before scaling. Diagnostic results must remain distinct from pure-Qwen/formal results.
- Laya selects concrete actions; Qwen supplies slow reasoning, evidence interpretation and semantic Monitor judgments. Programs validate, execute and save; they do not complete model-owned nodes.
- Every Qwen send requires a preflight including text, actual images and reserved output with a total ceiling of 32,768. Qwen2-VL surrogate tokenizers are not a verified tokenizer for qwen3.8-max; any conservative counting method and limitations must be explicit.
- Laya state capacity and per-option/head limits are now checked separately, before HTTP, using its actual downloaded tokenizer. No overlength contract is sent or silently truncated. This is an 8K ceiling, not a 32K Laya deployment or a verified long-context semantic benchmark.
- Preserve original B=10, R=6, T=17 meanings; freeze a mapping and charge/log all additional fast/slow model calls and output reservations before execution.
- Do not deploy/switch local Qwen2-VL-2B or 72B, provision another instance, or start additional GPU jobs.
- Snapshot provenance and frozen candidate manifests are separate from the main source checkout and all formal/pure-Qwen outputs.

## Download provenance

The server can access Hugging Face files through its pre-existing `/etc/network_turbo` configuration. Direct Hugging Face requests were unreachable; `hf-mirror.com` returned 429/403, and the Hub metadata request through the accelerated route returned 503. The deployment therefore downloads the exact necessary files from the official Hub at frozen revision `e333cb7c823cd1787276228260f56d02fbf5180b`, then serves local directories without network lookups.

Official Hub LFS metadata was checked read-only on 2026-11-19:

| Checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| english | 842609210 | `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` |
| multilingual | 643835514 | `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` |

`pip check` reports no broken requirements. Serving scripts pass `bash -n`. Weight verification and actual inference results are recorded below when complete.

## Acceptance results

- English example: `technical`, probabilities `{technical: 0.9691, billing: 0.0105, other: 0.0204}`; input_tokens=42, output_tokens=0, no truncation.
- Chinese example: `technical`, probabilities `{technical: 0.9651, billing: 0.0036, other: 0.0313}`; input_tokens=58, output_tokens=0, no truncation. This first Chinese request measured 530.88 ms end to end / 527.18 ms server inference and is not a throughput benchmark.
- Final health: english and multilingual both on cuda; device_is_preference=false; CPU fallback counters both 0. The service remains running after the short SSH commands disconnect.
- Final startup script idempotence check reports the existing PID, rather than starting a second server. Scripts pass bash -n; pip check reports no broken requirements.
- Default inference budgets: english max_len=512/head_max_len=192; multilingual max_len=1024/head_max_len=256. These are defaults, not absolute maxima. Both encoders declare max_position_embeddings=8192; the HTTP override ceiling is 8192. An English request with input_tokens=7924 fit without truncation. A longer probe was clamped to 8192, dropping 132 state tokens; max_len=8193 was rejected with HTTP 422. Repeated-word capacity tests establish retention accounting only, not long-context reasoning quality. Evidence: `outputs/laya-joint-20261119/capacity-validation.json`.
- Installed `laya.common.build_head` caps each rendered option at 48 tokens even with a larger head budget; instruction text can also be cut by the head budget. The joint client checks complete short option descriptions and instructions, keeps complete action arguments in state, and verifies returned usage against the exact preflight.
- The shipped English 11+ choice temperature is clamped by the official runtime, which warns that affected confidence values are uncalibrated. No checkpoint parameters were edited.
- Health revisions are null when serving local directories; use the separately recorded source revision and verified weight hashes to identify this deployment.
- Remote evidence: /root/autodl-tmp/laya-deploy/validation.json, example.json, example-result.json, health-final.json, weights.sha256 and README.zh-CN.md. Local copies of acceptance data and launch files: outputs/laya-deployment-20261119/ in this worktree only.

## Quick use on the server

```bash
cd /root/autodl-tmp/laya-deploy
curl -sS http://127.0.0.1:18080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @example.json | .venv/bin/python -m json.tool
```

Change state and criteria in example.json. Use model=multilingual for Chinese or model=english for English. The result is at answers.department.choice.

Start: bash /root/autodl-tmp/laya-deploy/start.sh. Stop: bash /root/autodl-tmp/laya-deploy/stop.sh. Logs: /root/autodl-tmp/laya-deploy/laya.log.

The service binds 127.0.0.1:18080 on the server; access from a laptop requires an SSH forward to this loopback port. SSH disconnection does not stop the service. Machine/container restart requires running start.sh again; automatic boot supervision is not configured.
