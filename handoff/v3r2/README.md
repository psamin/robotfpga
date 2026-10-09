# FPGA handoff v3r2

Int8 TinyPolicy weights and golden vectors for milestone M4 (#15), from the SW-16 pipeline (PR #58).

## Files (this folder)

- `weights.bin`: 568,240 bytes, SHA-256 `c87abfae0cbe…` (full hash in `manifest.json`)
- `manifest.json`: per-layer type, shapes, shift, byte offsets
- `vectors/`: 100 golden vectors, `NNN_in.bin` (27,658 bytes) and `NNN_out.bin` (48 bytes), plus per-layer dumps for the first 5
- `results/`: eval JSONs and the HLS C-model log

**HLS C model:** PASS on all 100 vectors, layer dumps on the first 5, 0 failures (`results/hls_csim.txt`).

## Provenance

Job 8986, `cluster/final.sbatch` at `32d24bd`. QAT (10,000 steps, seed 0) from the oblique-camera DAgger
round-2 float checkpoint `dagger-v3/round2/ckpt_030000.pt`, power-of-two scales, bit-exact with
`ref/intref.py`. The older v2 run (job 8983) scored 0% and is superseded.

## Closed-loop sim results

50 episodes per instruction for Stage B/C, 25 per instruction for the latency sweep; Wilson 95% CI.

| Backend | Stage | Injected latency | Success | 95% CI | Inference p50 |
|---|---|---|---|---|---|
| float (torch) | B | 0 ms | 199/200 = 100% | [0.97, 1.00] | 1.42 ms |
| float (torch) | C | 0 ms | 196/200 = 98% | [0.95, 0.99] | 1.36 ms |
| int8 (intref) | B | 0 ms | 182/200 = 91% | [0.86, 0.94] | 7.58 ms |
| int8 (intref) | C | 0 ms | 180/200 = 90% | [0.85, 0.93] | 7.71 ms |
| int8 (intref) | C | 33 ms | 86/100 = 86% | [0.78, 0.91] | 7.47 ms |
| int8 (intref) | C | 67 ms | 83/100 = 83% | [0.74, 0.89] | 7.50 ms |
| int8 (intref) | C | 100 ms | 84/100 = 84% | [0.76, 0.90] | 7.50 ms |
| int8 (intref) | C | 133 ms | 78/100 = 78% | [0.69, 0.85] | 7.65 ms |
| int8 (intref) | C | 200 ms | 64/100 = 64% | [0.54, 0.73] | 7.57 ms |
| int8 (intref) | C | 300 ms | 66/100 = 66% | [0.56, 0.75] | 7.54 ms |
| int8 (intref) | C | 500 ms | 51/100 = 51% | [0.41, 0.61] | 9.22 ms |

Int8 keeps 90% Stage C success against 98% for float at zero injected latency, and stays above 80% up to
about 100 ms. Inference times are the numpy `intref` model on the cluster CPU, not the board.
