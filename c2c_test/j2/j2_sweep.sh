#!/usr/bin/env bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# j2_sweep.sh -- JOB 2 head-to-head sweep driver.
# Builds and runs the pre-registered grid, at most MAXJOBS simulators at once.
# Usage: ./j2_sweep.sh <gridfile>
# gridfile lines: ARM CS TILE_KB REUSE
set -u
BASE=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd "$BASE/j2" || exit 1
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
export PATH=$BASE/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}

GRID=${1:-grid.txt}
MAXJOBS=${MAXJOBS:-4}
EPOCHS=${EPOCHS:-16}
NCL=${NCL:-4}
TP=${TP:-4}

running(){ jobs -rp | wc -l; }

while read -r ARM CS TKB RU; do
  [ -z "${ARM:-}" ] && continue
  case "$ARM" in \#*) continue;; esac
  tag="a${ARM}_cs${CS}_t${TKB}_r${RU}"
  d="run_$tag"
  # skip finished
  if [ -f "$d/run.log" ] && grep -q "^J2 " "$d/run.log" 2>/dev/null; then
    echo "SKIP $tag (done)"; continue
  fi
  bin="b_$tag"
  if [ ! -x "$bin" ]; then
    nvcc -arch=sm_90 ../j2_headtohead.cu -o "$bin" \
      -DARM=$ARM -DCS=$CS -DTILE_KB=$TKB -DREUSE=$RU \
      -DTILE_PASSES=$TP -DEPOCHS=$EPOCHS -DNCLUSTERS=$NCL -lcudart 2>build_$tag.err
    if [ ! -x "$bin" ]; then echo "BUILDFAIL $tag"; continue; fi
    patchelf --remove-rpath "$bin"
  fi
  mkdir -p "$d"; cp gpgpusim.config "$d/"; cp "$bin" "$d/"
  ( cd "$d" && ./"$bin" > run.log 2>&1 ) &
  echo "LAUNCH $tag"
  while [ "$(running)" -ge "$MAXJOBS" ]; do sleep 20; done
done < "$GRID"

wait
echo "SWEEP_COMPLETE"
