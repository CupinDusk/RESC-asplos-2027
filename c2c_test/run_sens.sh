#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
for LAT in 171 2000; do
  cp cfg_${LAT}.config gpgpusim.config
  echo "===== C2C SENS lat=$LAT cs=8 START $(date) ====="
  ./resc10240 cluster_size=8 repeat=1 num_clusters=200 avg_elems=128 burst_period=16 reuse_passes=4 > out_sens_${LAT}.log 2>&1
  echo "lat=$LAT EXIT=$? cycles=$(grep gpu_tot_sim_cycle out_sens_${LAT}.log | tail -1 | grep -oE "[0-9]+") shares=$(grep -c "Read share SUCCESS" out_sens_${LAT}.log)"
done
echo "SENS_DONE"
