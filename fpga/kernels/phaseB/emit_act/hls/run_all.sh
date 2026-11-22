#!/bin/sh
# The emit_act csynth set, one run at a time: waits while any vitis_hls.exe runs (other agents' runs included).
#   sh hls/run_all.sh     -> build/hls/*.log, build/hls/ea_<top>_<variant>_p<period>/
HERE="$(cd "$(dirname "$0")" && pwd)"
for cfg in "patched dq 4.67" "base dq 4.67" "patched dq 4.0" "base dq 4.0" "patched st 4.67" "base st 4.67"; do
  while :; do
    while tasklist 2>/dev/null | grep -qi vitis_hls.exe; do sleep 20; done
    out=$(sh "$HERE/hls_run.sh" $cfg)
    echo "$(date +%H:%M:%S) $out"
    case "$out" in *"not started"*) sleep 20 ;; *) break ;; esac
  done
done
echo "ALL DONE"
