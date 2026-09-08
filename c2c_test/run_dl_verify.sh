#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Verify: RESC completes AND is bit-identical-correct at nc up to 300 when the
# [RESC] printf flood is filtered. Compare RESC OUT_CHECKSUM vs global (correct baseline).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
# build global (correct baseline) + resc, no compute tile, pure comm
nvcc -arch=sm_90 resc_e2e.cu -o gl_v -DBACKEND=0 -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 -lcudart 2>/dev/null && patchelf --remove-rpath gl_v 2>/dev/null
echo "arm nc exit elapsed checksum"
for nc in 200 256 300 400; do
  for arm in gl_v resc_dl; do
    d=v_${arm}_${nc}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
    ( cd $d; t0=$(date +%s)
      timeout 700 ../$arm cluster_size=8 repeat=2 num_clusters=$nc avg_elems=128 burst_period=16 reuse_passes=2 2>/dev/null | grep -vE "\[RESC\]" > f.log
      ec=$?; t1=$(date +%s)
      cs=$(grep OUT_CHECKSUM f.log | head -1 | grep -oE "sum=[0-9.]+")
      echo "$arm $nc exit=$ec $((t1-t0))s $cs" )
  done
done
echo DL_VERIFY_DONE
