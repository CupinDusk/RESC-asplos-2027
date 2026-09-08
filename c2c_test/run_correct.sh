#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
# Small config, NO compute (CI=0) so out[] is PURE comm accumulation -> arms must match exactly.
# NC small, cluster 8, reuse 4. Compare OUT_CHECKSUM across global/dsm/resc.
W=correct_work; mkdir -p $W; rm -f $W/*
for arm in global dsm resc; do
  case $arm in
    global) be=0; extra="";;
    dsm) be=1; extra="";;
    resc) be=2; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
  esac
  nvcc -arch=sm_90 resc_e2e.cu -o $W/${arm}_c -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 $extra -lcudart 2>/dev/null
  patchelf --remove-rpath $W/${arm}_c 2>/dev/null
done
for arm in global dsm resc; do
  ( d=$W/r_$arm; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config; cd $d
    ../${arm}_c cluster_size=8 repeat=4 num_clusters=16 avg_elems=512 burst_period=16 reuse_passes=4 > run.log 2>&1 ) &
done
wait
echo "arm OUT_CHECKSUM"
for arm in global dsm resc; do
  cs=$(grep OUT_CHECKSUM $W/r_$arm/run.log 2>/dev/null | head -1)
  echo "$arm: $cs"
done
echo CORRECT_DONE
