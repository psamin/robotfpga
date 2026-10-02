#!/bin/bash
#SBATCH -J armlab-final
#SBATCH -p general
#SBATCH --gres=gpu:l4:1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
# Float checkpoint -> QAT -> export -> closed-loop eval (float vs int8, Stage B/C, latency sweep)
# -> golden vectors -> FPGA HLS C model. Everything lands in runs/$NAME.
#   sbatch --export=ALL,NAME=final-v3,FLOAT=<ckpt>,DATA="<dir> <dir> ..." cluster/final.sbatch
source cluster/env.sh
OUT=$(fix "$HOME/robotfpga-work/runs/${NAME:-final}")
FLOAT=$(fix "$FLOAT")
DATA=$(for d in $DATA; do fix "$d"; done | tr "\n" " ")
mkdir -p "$OUT"
W=${SLURM_CPUS_PER_TASK:-8}
EP=${EP:-50}
run() { echo "+ $*"; "$UV" run python -m "$@"; }

run armlab.policy.qat_train --data $DATA --data-device cpu --float-ckpt "$FLOAT" --out "$OUT/qat" ${QAT_ARGS:-}
for st in B C; do
  run armlab.eval.run --backend torch --ckpt "$FLOAT" --stage $st --episodes $EP --workers $W --out "$OUT/eval_float_$st.json"
  run armlab.eval.run --backend intref --ckpt "$OUT/qat/export" --stage $st --episodes $EP --workers $W --out "$OUT/eval_int8_$st.json"
done
run armlab.eval.run --backend intref --ckpt "$OUT/qat/export" --stage C --episodes 25 --workers $W \
  --latency-ms 33 67 100 133 200 300 500 --out "$OUT/eval_int8_C_latency.json"

# Golden vectors with the FPGA side's tools (PRs #6 and #8), taken from their branches read-only.
HW=$SCR/hw && mkdir -p "$HW"
git fetch -q origin ref-vectors hls-kernel
git archive origin/ref-vectors ref | tar -x -C "$HW"
git archive origin/hls-kernel hls | tar -x -C "$HW"
"$UV" run python scripts/pack_frames.py --data ${DATA%% *} --out "$OUT/packets.npy" --n 80
cp -r "$OUT/qat/export" "$OUT/handoff"
(cd "$HW" && "$UV" run --project "$REPO" python ref/vectors.py --model "$OUT/handoff" --packets "$OUT/packets.npy")
make -s -C "$HW/hls" csim ART="$OUT/handoff" | tee "$OUT/hls_csim.txt"
