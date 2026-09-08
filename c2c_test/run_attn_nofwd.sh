#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# What forwarding is worth on the flagship attention kernel.
# Prereg 794dcb8e509cd851ad7b96b8204b57ff0acab36aef7c9ff9acbc42d1abc8758a
#
# ONE cell, ONE env var apart from the already-run AT_RESC (131,665 cyc):
#   RESC_NO_FWD=1  disables marked-line cross-SM L1-to-L1 serve (gpu-cache.cc:278, LIVE file).
# Reuses the EXISTING dec_resc binary (md5 fcfd697021e8033eb77b8f64259b6382) -- no rebuild,
# so no rpath / nvcc-vintage risk. Same argv, same knob 3, same G=8, same dl2 S:960.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
W=$T/c2c_test/attn_a6
BIN=$W/bin
ARGS="4 8 8"          # POSITIONAL (resc_decode.cu:246-249)

# Guard: the binary must be the very one AT_RESC used.
WANT=fcfd697021e8033eb77b8f64259b6382
GOT=$(md5sum "$BIN/dec_resc" | awk '{print $1}')
if [ "$GOT" != "$WANT" ]; then echo "FATAL binary changed: $GOT != $WANT"; exit 1; fi
echo "BINARY OK $GOT"

nm=AT_NOFWD
d=$W/$nm
mkdir -p "$d"
# Copy AT_RESC's own config verbatim -- identical L2 and identical knob by construction.
cp "$W/AT_RESC/gpgpusim.config" "$d/gpgpusim.config"
grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960"; exit 1; }
grep -q "^-gpgpu_cluster_barrier_invalidate_l1 3\$" "$d/gpgpusim.config" || { echo "FATAL knob!=3"; exit 1; }
echo "CONFIG OK $(grep -m1 -oE 'dl2 S:[0-9]+' "$d/gpgpusim.config") knob=3"

rm -f "$d/exit.code"
cd "$d" || exit 1
RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 RESC_NO_FWD=1 \
  timeout 21600 nice -n 10 "$BIN/dec_resc" $ARGS > run.log 2>&1
echo $? > exit.code

echo "=== DONE ==="
L=$d/run.log
echo "exit=$(cat "$d/exit.code")"
grep -m1 "gpu_sim_cycle =" "$L"
grep -m1 "gpu_tot_sim_insn =" "$L"
grep -oE "OUT_CHECKSUM sum=[-0-9.]+" "$L" | head -1
grep -oE "readshare_(served|expired)=[0-9]+|owner_retained=[0-9]+" "$L" | tail -3
grep -oE "(whole|selective)_invalidates=[0-9]+" "$L" | tail -2
grep -c "\[RESC_NO_FWD\]" "$L" | sed 's/^/NO_FWD_banner_count=/'
