#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# The five RESC (knob 3) probe-port cells ONLY. PPA6 is already running under an earlier driver.
# Prereg be694aa96f9ccd40cc26fd3f88a64a20daeb087ce00249d2ae505a12d29ac84e
# FIX APPLIED: cx_arm2 requires RESC_RETAIN=1 RESC_RETAIN_Q=100 (run_composed.sh sets them for the
# A2 arm only); without them the binary exits before the kernel and writes exit=1 with no cycles.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
C=$T/c2c_test/composed
export PATH=$T/c2c_test/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
BIN=$C/bin
W=$C/sweep_probeport
ARGS="2 32 4 64 256"
MAXJOBS=3

echo "### sim .so md5: $(md5sum $T/build/lib/libcudart.so.12 | awk '{print $1}')"
echo "### cx_arm2 md5: $(md5sum $BIN/cx_arm2 | awk '{print $1}')"

running(){ ps -u hyren -o comm --no-headers | grep -c '^cx_arm'; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }

for pp in 0 1 2 4 8; do
  d=$W/PP$pp
  [ "$pp" = "0" ] && d=$W/PP0
  mkdir -p "$d"
  cp "$C/sweep_l2corrected/C60_L1P_A2/gpgpusim.config" "$d/gpgpusim.config"
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 PP$pp"; continue; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 3\$" "$d/gpgpusim.config" || { echo "FATAL knob PP$pp"; continue; }
  rm -f "$d/exit.code" "$d/run.log"
  throttle
  echo "LAUNCH PP$pp RESC_PROBE_PORT=$pp (retention ON)"
  ( cd "$d" && timeout 21600 nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 \
      RESC_RETAIN=1 RESC_RETAIN_Q=100 RESC_PROBE_PORT=$pp \
      "$BIN/cx_arm2" $ARGS > run.log 2>&1; echo $? > exit.code ) &
done
wait
echo "=== DONE ==="
for pp in 0 1 2 4 8; do
  d=$W/PP$pp; L=$d/run.log
  printf "PP%-3s exit=%-4s cyc=%-9s insn=%-11s %s retain=%s probe=%s\n" "$pp" \
    "$([ -f $d/exit.code ] && cat $d/exit.code || echo NONE)" \
    "$(grep -m1 'gpu_sim_cycle =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -m1 'gpu_tot_sim_insn =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -oE 'OUT_CHECKSUM[^ ]* sum=[-0-9.]+' $L 2>/dev/null | head -1)" \
    "$(grep -c '\[RESC_RETAIN\]' $L 2>/dev/null)" \
    "$(grep -c '\[RESC_PROBE_PORT\]' $L 2>/dev/null)"
done
