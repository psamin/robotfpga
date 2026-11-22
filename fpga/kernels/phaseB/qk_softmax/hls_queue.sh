#!/bin/sh
# Runs the csynth set one after another, each only when no other vitis_hls.exe runs on the machine:
#   tail (the fused tail alone), base (the integrator's GEMM unit, snapshot) and fused (the fused GEMM unit), at
#   QKS_PERIOD (default 4.67 ns); then fused at 4.0 ns if QKS_ALSO_40=1.
HERE="$(cd "$(dirname "$0")" && pwd)"
P=${QKS_PERIOD:-4.67}
wait_free() {
  while tasklist 2>/dev/null | grep -qi vitis_hls.exe; do sleep 30; done
}
for top in ${QKS_TOPS:-tail base fused}; do
  wait_free
  echo "$(date) start $top $P"
  QKS_TOP=$top QKS_PERIOD=$P sh "$HERE/hls_run.sh" "qks_${top}_p$(echo $P | tr -d .)"
  echo "$(date) end $top"
done
if [ "$QKS_ALSO_40" = "1" ]; then
  wait_free
  echo "$(date) start fused 4.0"
  QKS_TOP=fused QKS_PERIOD=4.0 sh "$HERE/hls_run.sh" qks_fused_p40
  echo "$(date) end fused 4.0"
fi
