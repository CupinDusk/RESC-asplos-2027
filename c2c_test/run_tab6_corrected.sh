#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Table 6 (the flagship attention sweep) on the CORRECTED 60 MiB L2 -- the r25 5/5 gate.
# Prereg 53926cc12f10ec35f30a1ec3d03dd1de99cccc8077d74258df4c8e2d6c91dc2a
#
# 18 cells = 6 compute scales x 3 arms. Argv "4 4 8" (POSITIONAL, resc_decode.cu:246-249)
#   -- NOT r24's "4 8 8". Published checksum for this shape is 2.096000 (guard T1).
# Recipe per ATTENTION_RESULTS.txt: CONDA nvcc 12.9 (system nvcc -> 0 registrations),
#   ptxas_shim on PATH, NO -L build/lib on the nvcc line, strip conda's injected rpath.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
CONDA_NVCC=$HOME/miniconda3/envs/clustersim-py311/bin/nvcc
SHIM=$(ls -d $T/c2c_test/ptxas_shim 2>/dev/null || true)
export PATH=${SHIM:+$SHIM:}$HOME/miniconda3/envs/clustersim-py311/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
SRC=$T/c2c_test/resc_decode.cu
W=$T/c2c_test/tab6_corrected
BIN=$W/bin
ARGS="4 4 8"
DL2_960='-gpgpu_cache:dl2 S:960:128:16,L:B:m:L:L,A:192:4,32:0,32'
MAXJOBS=6
mkdir -p "$W" "$BIN"

build() {
  local nm="$1"
  shift
  [ -x "$BIN/$nm" ] && { echo "SKIP build $nm"; return 0; }
  env NVCC_PREPEND_FLAGS= NVCC_APPEND_FLAGS= LDFLAGS= CFLAGS= CPPFLAGS= \
      "$CONDA_NVCC" -arch=sm_90 "$@" "$SRC" -o "$BIN/$nm" -lcudart 2>&1 | tail -3
  [ -x "$BIN/$nm" ] || { echo "BUILD FAILED $nm"; return 1; }
  if readelf -d "$BIN/$nm" | grep -qi "rpath\|runpath"; then
    echo "BUILD REJECTED $nm (rpath -> would load the real libcudart)"; rm -f "$BIN/$nm"; return 1
  fi
  echo "BUILT $nm"
}

build t6_global -DBACKEND=0                              || exit 1
build t6_dsm    -DBACKEND=1                              || exit 1
build t6_resc   -DBACKEND=2 -DREAD_MODE=1 -DWRITE_MODE=1 || exit 1

# $1=cell $2=binary $3=knob $4=base cfg
launch() {
  local nm="$1"
  local b="$2"
  local knob="$3"
  local basecfg="$4"
  local d="$W/$nm"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm"; return; fi
  mkdir -p "$d"
  awk -v l2="$DL2_960" -v k="$knob" '
    /^-gpgpu_cache:dl2 / { print l2; next }
    /^-gpgpu_cluster_barrier_invalidate_l1 / { print "-gpgpu_cluster_barrier_invalidate_l1 " k; next }
    { print }' "$basecfg" > "$d/gpgpusim.config"
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || \
    echo "-gpgpu_cluster_barrier_invalidate_l1 $knob" >> "$d/gpgpusim.config"
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 $nm"; return; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL knob $nm"; return; }
  rm -f "$d/exit.code"
  (
    cd "$d" || exit 1
    RESC_QUIET=1 RESC_MW=1 timeout 21600 nice -n 10 "$BIN/$b" $ARGS > run.log 2>&1
    echo $? > exit.code
  ) &
  echo "LAUNCHED $nm knob=$knob cfg=$(basename "$basecfg")"
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n 2>/dev/null || sleep 5; done
}

for s in 1_0 2_0 3_0 3_2 4_0 5_0; do
  C=$T/c2c_test/cfg_cs$s.config
  [ -f "$C" ] || { echo "FATAL missing $C"; continue; }
  launch "T6_${s}_global" t6_global 0 "$C"
  launch "T6_${s}_dsm"    t6_dsm    1 "$C"
  launch "T6_${s}_resc"   t6_resc   3 "$C"
done
wait

echo "=== DONE ==="
for s in 1_0 2_0 3_0 3_2 4_0 5_0; do
  for a in global dsm resc; do
    d=$W/T6_${s}_${a}; L=$d/run.log
    ec=$( [ -f "$d/exit.code" ] && cat "$d/exit.code" || echo NONE )
    cyc=$(grep -m1 "gpu_sim_cycle =" "$L" 2>/dev/null | awk '{print $3}')
    insn=$(grep -m1 "gpu_tot_sim_insn =" "$L" 2>/dev/null | awk '{print $3}')
    ck=$(grep -oE "OUT_CHECKSUM sum=[-0-9.]+" "$L" 2>/dev/null | head -1)
    wi=$(grep -oE "whole_invalidates=[0-9]+" "$L" 2>/dev/null | tail -1)
    si=$(grep -oE "selective_invalidates=[0-9]+" "$L" 2>/dev/null | tail -1)
    l2=$(grep -m1 -oE "dl2 S:[0-9]+" "$d/gpgpusim.config" 2>/dev/null)
    k=$(grep -oE "^-gpgpu_cluster_barrier_invalidate_l1 [0-9]+" "$d/gpgpusim.config" 2>/dev/null | grep -oE "[0-9]+$")
    echo "T6 $s $a exit=$ec cyc=$cyc insn=$insn $ck $wi $si $l2 knob=$k"
  done
done
