# Original Qwen thinking run resumed — 2026-10-22

The user explicitly changed the active task to resume the earlier Qwen run, rather than continue the newer inference experiments. The newer Qwen full run, Qwen validation and remaining GPT experimental runs were stopped by verified process-group identity. Their outputs were preserved; see `new_experiments_stopped.json`.

The original output is `/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs/v74_r8_r18_thinking_full7805_20261021`. It contained **2,889** recorded outcomes before resumption: 1,972 correct, 18 recorded errors and 4,916 queries remaining out of 7,805. Its completed-fragment accuracy was 68.26%; this is not a completed full-benchmark score.

All **152 inference source files** were recovered from `outputs/r13_single_pass_20261020/source_r18/local` and matched individually against the original run's `manifest.json`. The recovered files are isolated in `runtime_original/local`; no newer candidate fallback, observation compaction, image limit, loop fix or spatial-order experiment was inserted into this runtime. Its source manifest records every hash. The current working experiment code is not used by the resume launcher.

The original API and policy configuration were restored exactly: `qwen3.7-plus`, `enable_thinking=true`, an 8,192-token minimum, temperature zero, native JSON-schema output, `max_completion_tokens`, Top-5 categories, eight workers and concurrency eight. The original `provider=openai` value names its compatibility adapter; the actual model remains Qwen. Both endpoint and model match the original manifest. The updated credential is read from the private Token Plan profile outside Git.

The original driver accepted `--prepare-only` in the existing output directory without changing the original manifest: 2,889 cached outcomes and 4,916 pending. All prior evaluation hashes were checked unchanged. An image request through the exact original API configuration then succeeded, returned `qwen3.7-plus`, and reported reasoning tokens; see `original_provider_health.json` and `resume_preflight.json`.

`resume_original_qwen.sh` resumes this same directory. Existing correct, incorrect and failed evaluations are reused equally; only queries without evaluations continue. No result from any newer run is imported. `before_resume.json` inventories the existing evaluations and prediction-file metadata, and `resume_launch.json` identifies the resumed process. The original manifest and all prior outcomes must remain unchanged during continuation.

The original full run is still incomplete. Its final Overall and any 65–66% target require completion of all 7,805 queries with failures and abstentions retained in the denominator.
