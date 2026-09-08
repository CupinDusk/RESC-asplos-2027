#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MD real kernel on the corrected 60 MiB L2 -- the 5/5 reviewer objection.
# Prereg 7b4abc3dc4ddb45668c3a9ce63241a1b8c264e12ef6b291cc03802def1ad2938
# Controls (S:64) must reproduce the published 68,537 / 113,671 before the S:960 pair is read.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
SRC=$T/c2c_test/resc_md.cu
W=$T/c2c_test/md_l2corrected
BIN=$W/bin
BASECFG=$T/c2c_test/cfg_h20.config   # the config run_md_sweep.sh:13 actually copies
ARGS="num_clusters=16 repeat=2 pbar=16"    # AUTHORITATIVE: run_md_sweep.sh (md5 ae166ffd96e8)
                                            # line 9 NC=16 RPT=2, line 11 pbar loop 16/32/48;
                                            # the published 68,537 / 113,671 are the pbar=16 rows
DL2_960='-gpgpu_cache:dl2 S:960:128:16,L:B:m:L:L,A:192:4,32:0,32'
MAXJOBS=6
mkdir -p "$W" "$BIN"

# ---- build, in a purged env with the SYSTEM nvcc (D6): conda injects -Wl,-rpath ----
build() {
  local nm="$1"; shift
  [ -x "$BIN/$nm" ] && { echo "SKIP build $nm"; return 0; }
  env -u CONDA_PREFIX -u LD_LIBRARY_PATH \
      NVCC_PREPEND_FLAGS= NVCC_APPEND_FLAGS= LDFLAGS= CFLAGS= CPPFLAGS= \
      PATH=/usr/local/cuda-12.4/bin:/usr/bin:/bin \
      /usr/local/cuda-12.4/bin/nvcc -arch=sm_90 "$@" "$SRC" -o "$BIN/$nm" -lcudart 2>&1 | tail -3
  [ -x "$BIN/$nm" ] || { echo "BUILD FAILED $nm"; return 1; }
  if readelf -d "$BIN/$nm" | grep -qi "rpath\|runpath"; then
    echo "BUILD REJECTED $nm (carries rpath)"; rm -f "$BIN/$nm"; return 1
  fi
  echo "BUILT $nm"
}

build md_cg   -DCLUSTER_SIZE=8 -DBACKEND=0 -DRESERVE_ELEMS=8192                             || exit 1
build md_resc -DCLUSTER_SIZE=8 -DBACKEND=2 -DRESERVE_ELEMS=8192 -DREAD_MODE=1 -DWRITE_MODE=1 || exit 1

running_sims() { pgrep -u hyren -f "md_cg|md_resc" | wc -l; }
throttle() { while [ "$(running_sims)" -ge "$MAXJOBS" ]; do sleep 30; done; }

# $1=cell $2=binary $3=l2 ("64"=keep base, "960"=swap)
launch() {
  local nm="$1" b="$2" l2="$3"
  local d="$W/$nm"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm"; return; fi
  for p in $(pgrep -u hyren -f "md_cg|md_resc"); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && { echo "IN-FLIGHT $nm"; return; }
  done
  throttle
  mkdir -p "$d"                                  # D5: run dirs are PRESERVED this time
  if [ "$l2" = "960" ]; then
    awk -v newl="$DL2_960" '/^-gpgpu_cache:dl2 / { print newl; next } { print }' "$BASECFG" > "$d/gpgpusim.config"
    grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL swap failed $nm"; return; }
  else
    cp "$BASECFG" "$d/gpgpusim.config"
    grep -q "dl2 S:64" "$d/gpgpusim.config" || { echo "FATAL base not S:64 $nm"; return; }
  fi
  rm -f "$d/exit.code"
  (
    cd "$d" || exit 1
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    timeout 21600 nice -n 10 "$BIN/$b" $ARGS > run.log 2>&1
    echo $? > exit.code
  ) &
  echo "LAUNCHED $nm bin=$b l2=S:$l2"
}

launch MD64_CG    md_cg   64
launch MD64_RESC  md_resc 64
launch MD960_CG   md_cg   960
launch MD960_RESC md_resc 960

wait
echo "=== DONE ==="
for nm in MD64_CG MD64_RESC MD960_CG MD960_RESC; do
  d=$W/$nm
  ec=$( [ -f "$d/exit.code" ] && cat "$d/exit.code" || echo NONE )
  cyc=$(grep -m1 "gpu_sim_cycle =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  insn=$(grep -m1 "gpu_tot_sim_insn =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  ck=$(grep -o "OUT_CHECKSUM sum=[0-9.]*" "$d/run.log" 2>/dev/null | head -1)
  l2=$(grep -o "dl2 S:[0-9]*" "$d/gpgpusim.config" 2>/dev/null | head -1)
  echo "MDL2 $nm exit=$ec cyc=$cyc insn=$insn $ck | $l2"
done
