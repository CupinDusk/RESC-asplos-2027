#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# CORRECTNESS PROOF: does ld.ca (non-coherent L1 cache) read STALE data across
# epochs while ld.cg(global) and RESC stay correct? Many rounds, high reuse,
# compare OUT_CHECKSUM. If noncoh != global==resc, ld.ca is INCORRECT (not a method).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
W=stale_work; mkdir -p $W; rm -f $W/*
for arm in global noncoh resc; do
  case $arm in
    global) be=0; extra="";;
    noncoh) be=4; extra="";;
    resc) be=2; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
  esac
  nvcc -arch=sm_90 resc_e2e.cu -o $W/${arm}_s -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 $extra -lcudart 2>/dev/null
  patchelf --remove-rpath $W/${arm}_s 2>/dev/null
done
# High round count (many epochs = many overwrites), high reuse, SMALL avg_elems so
# addresses repeat across epochs (max cross-epoch staleness exposure).
for arm in global noncoh resc; do
  ( d=$W/r_$arm; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config; cd $d
    ../${arm}_s cluster_size=8 repeat=20 num_clusters=8 avg_elems=64 burst_period=3 reuse_passes=4 > run.log 2>&1 ) &
done
wait
echo "arm OUT_CHECKSUM (global==resc correct; noncoh divergent => ld.ca reads STALE => not a valid method)"
for arm in global noncoh resc; do
  cs=$(grep OUT_CHECKSUM $W/r_$arm/run.log 2>/dev/null | head -1)
  echo "$arm: $cs"
done
echo STALE_DONE
