#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Probe-port charge sweep, RE-HOSTED on maxunion (cx_arm2 does not execute; see void record).
# Parent prereg be694aa96f9ccd40cc26fd3f88a64a20daeb087ce00249d2ae505a12d29ac84e
# Addendum 01 (this) -- baselines from the discriminator run 91e283fb...:
#   mu_arm2 knob 3 be0 = 961,832 cyc / insn 65,044,352 / BITIDENTICAL=1
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=$T/c2c_test/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
BIN=$T/c2c_test/maxunion/bin
SRCCFG=$T/c2c_test/maxunion/sweep/D_ARM2_be0_pv256/gpgpusim.config
W=$T/c2c_test/maxunion/sweep_probeport
mkdir -p "$W"
ARGS="2 32 0 64 256"
MAXJOBS=3

[ -f "$SRCCFG" ] || { echo "FATAL missing $SRCCFG"; exit 1; }
echo "### sim .so md5: $(md5sum $T/build/lib/libcudart.so.12 | awk '{print $1}')  (patched; pre-patch 0a27b8aac382e749c66569ff706014f0)"
echo "### mu_arm2 md5: $(md5sum $BIN/mu_arm2 | awk '{print $1}')"

running(){ ps -u hyren -o comm --no-headers | grep -c '^mu_arm'; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }

for pp in 0 1 2 4 8; do
  d=$W/MPP$pp
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP MPP$pp"; continue; fi
  mkdir -p "$d"
  cp "$SRCCFG" "$d/gpgpusim.config"
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 MPP$pp"; continue; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 3\$" "$d/gpgpusim.config" || { echo "FATAL knob MPP$pp"; continue; }
  rm -f "$d/exit.code" "$d/run.log"
  throttle
  echo "LAUNCH MPP$pp RESC_PROBE_PORT=$pp"
  ( cd "$d" && timeout 21600 nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
      RESC_RETAIN=1 RESC_RETAIN_Q=100 RESC_PROBE_PORT=$pp \
      "$BIN/mu_arm2" $ARGS > run.log 2>&1; echo $? > exit.code ) &
done
wait

echo "=== DONE ==="
for pp in 0 1 2 4 8; do
  d=$W/MPP$pp; L=$d/run.log
  printf "MPP%-3s exit=%-4s cyc=%-9s insn=%-11s bitid=%s banner=%s served=%s\n" "$pp" \
    "$([ -f $d/exit.code ] && cat $d/exit.code || echo NONE)" \
    "$(grep -m1 'gpu_sim_cycle =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -m1 'gpu_tot_sim_insn =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -oE 'BITIDENTICAL=[01]' $L 2>/dev/null | tail -1)" \
    "$(grep -c '\[RESC_PROBE_PORT\]' $L 2>/dev/null)" \
    "$(grep -oE 'readshare_served=[0-9]+' $L 2>/dev/null | tail -1)"
done
