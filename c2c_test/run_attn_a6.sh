#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# The ld.ca+acquire arm on the flagship attention kernel -- r23 gpu-arch (f).
# Prereg 4da39dc612055c9b5489b1a8f9caf0c8a02dd9b19e2f96382d16a46ba4ae5ec3
#
# RECIPE RECOVERED FROM ATTENTION_RESULTS.txt (not invented):
#   * build with the CONDA nvcc (CUDA 12.9). The system nvcc 12.4/13.0 makes binaries whose
#     RESC registration reaches "Converted to global address" then exits with 0 registrations.
#   * PATH must include ptxas_shim.   * NO -L build/lib on the nvcc line (it links automatically).
#   * params: cluster=4 repeat=8 ncl=8   (the CORRECTNESS GATE line of that record)
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
CONDA_NVCC=$HOME/miniconda3/envs/clustersim-py311/bin/nvcc
SHIM=$(ls -d $T/c2c_test/ptxas_shim 2>/dev/null || true)
export PATH=${SHIM:+$SHIM:}$HOME/miniconda3/envs/clustersim-py311/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
SRC=$T/c2c_test/resc_decode.cu
W=$T/c2c_test/attn_a6
BIN=$W/bin
BASECFG=$T/c2c_test/cfg_h20.config
ARGS="4 8 8"          # POSITIONAL (resc_decode.cu:246-249): <cluster> <repeat> <ncl>
                       # = cluster=4 repeat=8 ncl=8, the CORRECTNESS GATE line of ATTENTION_RESULTS.txt
DL2_960='-gpgpu_cache:dl2 S:960:128:16,L:B:m:L:L,A:192:4,32:0,32'
mkdir -p "$W" "$BIN"

build() {
  local nm="$1"; shift
  [ -x "$BIN/$nm" ] && { echo "SKIP build $nm"; return 0; }
  # Keep the CONDA nvcc (required for registration) but strip conda's injected
  # -Wl,-rpath: it pins the REAL libcudart ahead of LD_LIBRARY_PATH and the run
  # dies with "no kernel image is available". The working published dec_* have NO RPATH.
  env NVCC_PREPEND_FLAGS= NVCC_APPEND_FLAGS= LDFLAGS= CFLAGS= CPPFLAGS= \
      "$CONDA_NVCC" -arch=sm_90 "$@" "$SRC" -o "$BIN/$nm" -lcudart 2>&1 | tail -3
  [ -x "$BIN/$nm" ] || { echo "BUILD FAILED $nm"; return 1; }
  if readelf -d "$BIN/$nm" | grep -qi "rpath\|runpath"; then
    echo "BUILD REJECTED $nm (carries rpath -- would load the real libcudart)"; rm -f "$BIN/$nm"; return 1
  fi
  echo "BUILT $nm ($($CONDA_NVCC --version | tail -1 | cut -c1-40))"
}

build dec_resc -DBACKEND=2 -DREAD_MODE=1 -DWRITE_MODE=1 || exit 1
build dec_cg   -DBACKEND=0                              || exit 1

# $1=cell $2=binary $3=invalidate knob
launch() {
  local nm="$1" b="$2" knob="$3"
  local d="$W/$nm"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm"; return; fi
  mkdir -p "$d"
  awk -v l2="$DL2_960" -v k="$knob" '
    /^-gpgpu_cache:dl2 / { print l2; next }
    /^-gpgpu_cluster_barrier_invalidate_l1 / { print "-gpgpu_cluster_barrier_invalidate_l1 " k; next }
    { print }' "$BASECFG" > "$d/gpgpusim.config"
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 $nm"; return; }
  if ! grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config"; then
    echo "-gpgpu_cluster_barrier_invalidate_l1 $knob" >> "$d/gpgpusim.config"
  fi
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL knob $nm"; return; }
  rm -f "$d/exit.code"
  (
    cd "$d" || exit 1
    RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 \
    timeout 21600 nice -n 10 "$BIN/$b" $ARGS > run.log 2>&1
    echo $? > exit.code
  ) &
  echo "LAUNCHED $nm bin=$b knob=$knob"
}

launch AT_RESC dec_resc 3     # control: marked-selective
launch AT_A6   dec_resc 1     # THE MISSING ARM: same binary, whole-L1 acquire
launch AT_CG   dec_cg   0     # the ld.cg global arm

wait
echo "=== DONE ==="
for nm in AT_RESC AT_A6 AT_CG; do
  d=$W/$nm
  ec=$( [ -f "$d/exit.code" ] && cat "$d/exit.code" || echo NONE )
  cyc=$(grep -m1 "gpu_sim_cycle =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  insn=$(grep -m1 "gpu_tot_sim_insn =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  ck=$(grep -o "OUT_CHECKSUM sum=[0-9.]*" "$d/run.log" 2>/dev/null | head -1)
  wi=$(grep -o "whole_invalidates=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  si=$(grep -o "selective_invalidates=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  reg=$(grep -c "CLUSTER_STATE" "$d/run.log" 2>/dev/null)
  k=$(grep -o "^-gpgpu_cluster_barrier_invalidate_l1 [0-9]*" "$d/gpgpusim.config" | grep -o "[0-9]*$")
  echo "ATTN $nm exit=$ec cyc=$cyc insn=$insn $ck $wi $si regs=$reg knob=$k"
done
