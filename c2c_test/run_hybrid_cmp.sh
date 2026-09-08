#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
# ensure arms exist: global10240/dsm10240/resc10240 already built. hybrid10240 just built (RESERVE=4096=16KB).
# NOTE: RESC and others were built with RESERVE_ELEMS=1 in the sweep; rebuild all 4 at SAME RESERVE for fair SMEM.
echo "arm payloadElems N cyc"
for ae in 128 1024 4096 8192 16384; do
  for cs in 2 8; do
    N=$([ $cs -eq 2 ] && echo 1 || echo 7)
    for arm in global dsm resc hybrid; do
      c=$(timeout 300 ./${arm}10240 cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$ae burst_period=16 reuse_passes=4 2>/dev/null | grep gpu_tot_sim_cycle | tail -1 | grep -oE "[0-9]+")
      echo "$arm $ae $N $c"
    done
  done
done
echo "HYBRID_CMP_DONE"
