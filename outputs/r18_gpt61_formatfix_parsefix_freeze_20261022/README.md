# R18 GPT-6.1 parse-fix freeze

This snapshot freezes the predicted-category Top-5, non-cumulative R18 pipeline after adding conservative relay-format recovery. The model still must satisfy the local schemas and evidence checks; malformed scene prose is allowed to become an abstention, never fabricated evidence.

- Code: `reproductions/SeeGround/local/` on branch `codex/r18-gpt61-formatfix-parsefix-freeze-20261022`.
- Predicted-category input: `/Users/luocl/Desktop/seeground/outputs/r8_grounding_input_20261020/20261020_r8_object_predictions`.
- Validation selection: `validation192_manifest.json`; retry selections are recorded beside it.
- Smoke: `smoke8_summary.json`.
- Merged validation audit: `validation192_merged_summary.json` and `validation192_merged_rows.json`.
- API credentials stay in `/Users/luocl/.config/seeground/gpt61-relay.env` and are not committed.

The full 7805 run remains a separate production job. This freeze records the exact code, selection manifests, input hash, and validation protocol needed to reproduce it.
