#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# L1 geometry + replacement-policy sensitivity at the headline point.
# Prereg 86a9e594577379bc5a14895d7ca3a3f5e893e974a451386f1df5e09b78a11007
# Derived from source: shader.cc:3601-3620 scales assoc from max_assoc (the dl1 S: field);
# gpu-cache.h:651-655 accepts only L(LRU) and F(FIFO) as replacement policy.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
W=$T/c2c_test/composed/sweep_l1geom
BINDIR=$T/c2c_test/composed/bin/v124
SRC=$T/c2c_test/composed/sweep_l2corrected     # corrected 60 MiB L2 cells
ARGS="2 32 4 64 256"                            # headline point: 8KB comm, 32KB private
MAXJOBS=6
mkdir -p "$W"

running_sims() { pgrep -u hyren -f "cx_arm" | wc -l; }
throttle() { while [ "$(running_sims)" -ge "$MAXJOBS" ]; do sleep 30; done; }

# $1=cellname $2=binary $3=source-config-dir $4=new dl1 line
launch() {
  local nm="$1" arm="$2" srcdir="$3" dl1="$4"
  local d="$W/$nm"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm"; return; fi
  for p in $(pgrep -u hyren -f "cx_arm"); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && { echo "IN-FLIGHT $nm pid=$p"; return; }
  done
  throttle
  mkdir -p "$d"
  # rewrite ONLY the active dl1 line; keep every other line byte-identical
  awk -v newl="$dl1" '/^-gpgpu_cache:dl1 / { print newl; next } { print }' \
      "$srcdir/gpgpusim.config" > "$d/gpgpusim.config"
  grep -q "^$dl1" "$d/gpgpusim.config" || { echo "FATAL dl1 swap failed $nm"; return; }
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 $nm"; return; }
  rm -f "$d/exit.code"
  (
    cd "$d" || exit 1
    RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 \
    timeout 21600 nice -n 10 "$BINDIR/$arm" $ARGS > run.log 2>&1
    echo $? > exit.code
  ) &
  echo "LAUNCHED $nm arm=$arm dl1='$dl1'"
}

PUB='-gpgpu_cache:dl1 S:4:128:64,L:T:m:L:L,A:512:64,16:0,32'
G16='-gpgpu_cache:dl1 S:16:128:16,L:T:m:L:L,A:512:64,16:0,32'
G32='-gpgpu_cache:dl1 S:32:128:8,L:T:m:L:L,A:512:64,16:0,32'
FIFO='-gpgpu_cache:dl1 S:4:128:64,F:T:m:L:L,A:512:64,16:0,32'

# Control pair first (P1): must reproduce 1,227,505 / 2,273,856 instruction-identically.
launch G4_A2  cx_arm2 "$SRC/C60_L1P_A2" "$PUB"
launch G4_A6  cx_arm6 "$SRC/C60_L1P_A6" "$PUB"
launch G16_A2 cx_arm2 "$SRC/C60_L1P_A2" "$G16"
launch G16_A6 cx_arm6 "$SRC/C60_L1P_A6" "$G16"
launch G32_A2 cx_arm2 "$SRC/C60_L1P_A2" "$G32"
launch G32_A6 cx_arm6 "$SRC/C60_L1P_A6" "$G32"
launch FF_A2  cx_arm2 "$SRC/C60_L1P_A2" "$FIFO"
launch FF_A6  cx_arm6 "$SRC/C60_L1P_A6" "$FIFO"

wait
echo "=== DONE ==="
for nm in G4_A2 G4_A6 G16_A2 G16_A6 G32_A2 G32_A6 FF_A2 FF_A6; do
  d=$W/$nm
  ec=$( [ -f "$d/exit.code" ] && cat "$d/exit.code" || echo NONE )
  cyc=$(grep -m1 "gpu_sim_cycle =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  insn=$(grep -m1 "gpu_tot_sim_insn =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  bit=$(grep -o "BITIDENTICAL=[01]" "$d/run.log" 2>/dev/null | head -1)
  rec=$(grep -o "Reconfigure L1 cache to [0-9]*KB" "$d/run.log" 2>/dev/null | head -1)
  dl1=$(grep -o "^-gpgpu_cache:dl1 S:[0-9]*:[0-9]*:[0-9]*,[A-Z]" "$d/gpgpusim.config" 2>/dev/null | head -1)
  echo "L1GEOM $nm exit=$ec cyc=$cyc insn=$insn $bit | $rec | $dl1"
done
