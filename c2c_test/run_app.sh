#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Probe-port charge on the FLAGSHIP ATTENTION kernel -- the host that actually serves.
# Parent prereg be694aa96f9ccd40cc26fd3f88a64a20daeb087ce00249d2ae505a12d29ac84e
# Addendum 02: f-hash in probe-port-charge-prereg-addendum-02.sha256
# Prior voids: composed cx_arm2 unrunnable (7e6513fa); maxunion readshare_served=0 (3b60ca7f).
# Baseline AT_RESC = 131,665 cyc / insn 17,735,896 / checksum 5.088000 / served 3,392.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
BIN=$T/c2c_test/attn_a6/bin
SRCCFG=$T/c2c_test/attn_a6/AT_RESC/gpgpusim.config
W=$T/c2c_test/attn_a6/sweep_probeport
ARGS="4 8 8"          # POSITIONAL (resc_decode.cu:246-249)
MAXJOBS=3
WANT_BIN=fcfd697021e8033eb77b8f64259b6382

[ -f "$SRCCFG" ] || { echo "FATAL missing $SRCCFG"; exit 1; }
GOT=$(md5sum "$BIN/dec_resc" | awk '{print $1}')
[ "$GOT" = "$WANT_BIN" ] || { echo "FATAL binary drift: $GOT != $WANT_BIN"; exit 1; }
mkdir -p "$W"
echo "### sim .so md5: $(md5sum $T/build/lib/libcudart.so.12 | awk '{print $1}')  (patched)"
echo "### dec_resc md5: $GOT (reused, no rebuild)"

running(){ ps -u hyren -o comm --no-headers | grep -c '^dec_resc'; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }

for pp in 0 1 2 4 8; do
  d=$W/APP$pp
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP APP$pp"; continue; fi
  mkdir -p "$d"
  cp "$SRCCFG" "$d/gpgpusim.config"
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 APP$pp"; continue; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 3\$" "$d/gpgpusim.config" || { echo "FATAL knob APP$pp"; continue; }
  rm -f "$d/exit.code" "$d/run.log"
  throttle
  echo "LAUNCH APP$pp RESC_PROBE_PORT=$pp"
  ( cd "$d" && timeout 21600 nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 \
      RESC_PROBE_PORT=$pp "$BIN/dec_resc" $ARGS > run.log 2>&1; echo $? > exit.code ) &
done
wait

echo "=== DONE ==="
for pp in 0 1 2 4 8; do
  d=$W/APP$pp; L=$d/run.log
  printf "APP%-3s exit=%-4s cyc=%-8s insn=%-11s %s banner=%s served=%s\n" "$pp" \
    "$([ -f $d/exit.code ] && cat $d/exit.code || echo NONE)" \
    "$(grep -m1 'gpu_sim_cycle =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -m1 'gpu_tot_sim_insn =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -oE 'OUT_CHECKSUM sum=[-0-9.]+' $L 2>/dev/null | head -1)" \
    "$(grep -c '\[RESC_PROBE_PORT\]' $L 2>/dev/null)" \
    "$(grep -oE 'readshare_served=[0-9]+' $L 2>/dev/null | tail -1 | cut -d= -f2)"
done
