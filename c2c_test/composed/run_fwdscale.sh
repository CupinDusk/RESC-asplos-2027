#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Forwarding-scaling cell: same as BIG_A2 but RESC_GEN_RING=3 (our published default window).
# Prereg 826f97584e330626daba0dcc32415f9386ef048b60e5055fd58f2e4365682f48
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
W=$T/c2c_test/composed/sweep_bigcomm
BINDIR=$T/c2c_test/composed/bin/v124
ARGS="2 32 4 2048 256"
nm=BIGW3_A2
d=$W/$nm

# Guard: never reuse a stale cell.
if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm already complete"; exit 0; fi
for p in $(pgrep -u hyren -f "cx_arm"); do
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && { echo "IN-FLIGHT $nm pid=$p"; exit 0; }
done

mkdir -p "$d"
# Copy BIG_A2's own run-directory config verbatim: identical config is a guard requirement (S5).
cp "$W/BIG_A2/gpgpusim.config" "$d/gpgpusim.config"
grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960"; exit 1; }
grep -q "cluster_barrier_invalidate_l1 3" "$d/gpgpusim.config" || { echo "FATAL wrong knob"; exit 1; }
rm -f "$d/exit.code"
(
  cd "$d" || exit 1
  RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
  timeout 21600 nice -n 10 "$BINDIR/cx_arm2" $ARGS > run.log 2>&1
  echo $? > exit.code
) &
echo "LAUNCHED $nm G=3 (BIG_A2 counterpart at G=8)"
sleep 3
for p in $(pgrep -u hyren -f "cx_arm"); do
  echo "  live pid=$p cwd=$(readlink /proc/$p/cwd 2>/dev/null)"
done
