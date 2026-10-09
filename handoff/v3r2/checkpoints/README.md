# v3r2 training checkpoints

The PyTorch checkpoints behind `handoff/v3r2/weights.bin`, copied from the GT cluster on 2026-10-09.

| File | What | Cluster source | SHA-256 |
|---|---|---|---|
| `float_dagger-v3_round2_ckpt_030000.pt` | Float TinyPolicy, EMA weights. 100/100 Stage B in sim | `~/robotfpga-work/runs/dagger-v3/round2/ckpt_030000.pt` | `8953d22ab205a5c745a0d959687cb322b873385a04955c2375efb407ee0dd1d9` |
| `qat_ckpt_qat.pt` | QAT model: float shadow weights + per-layer exponents (`a_exp`, `w_exp`) | `~/robotfpga-work/runs/final-v3r2/qat/ckpt_qat.pt` | `0f7e74decfcb7b9f3473a91179c5f1aae206befd78df506e7c555f853bd0f597` |
| `qat_meta.json` | QAT run metadata: config, git hash, seed, log, exponents, shifts, export SHA-256 | `~/robotfpga-work/runs/final-v3r2/qat/qat_meta.json` | — |

`qat_meta.json` records export SHA-256 `c87abfae0cbedf771a9ef927f39302a43cdcbb954f7090aca4baa6bef350a732`, the same as `../weights.bin`.

Load them like this:
- float: `TinyPolicy().load_state_dict(torch.load(path)["model"])`
- QAT: `armlab.policy.qat.load_qat(path)`

To redo QAT (for example, to close the int8 gap):
`python -m armlab.policy.qat_train --float-ckpt <float ckpt> --data <dataset dirs> --out <dir>`.
Run it on the cluster through `cluster/qat.sbatch` or `cluster/final.sbatch`.
