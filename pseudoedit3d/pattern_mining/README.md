# AML Pattern Mining Explorer v1 Package

This package is the new home for reusable Pattern Mining Explorer code.

Golden path:

```text
motion evidence extraction
-> candidate pattern mining
-> candidate audit
-> pattern registry
```

Current status: the v5 extractor still lives in
`scripts/audit_hml3d_multichannel_motion_bpe.py` for reproducibility. Do not add
new v6/v7 logic there. New extractor/miner/audit/registry modules should be
split into the subpackages below.

- `evidence_extractors/`: raw-joint, Layer3, and sidecar evidence emitters.
- `candidate_mining/`: coactivation, closed itemset, and optional channel-BPE miners.
- `pattern_audit/`: split-axis, phase, naming, TMR, and pseudo-GT diagnostics.
- `registry/`: accepted/component/split-required/blocked pattern registry export.
