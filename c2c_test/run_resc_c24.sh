#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_171.config gpgpusim.config
for cs in 2 4; do
  echo "===== resc cs=$cs START $(date) ====="
  ./resc10240 cluster_size=$cs repeat=1 num_clusters=200 avg_elems=128 burst_period=16 reuse_passes=4 > out_matrix_resc_c${cs}.log 2>&1
  echo "resc cs=$cs EXIT=$? cycles=$(grep gpu_tot_sim_cycle out_matrix_resc_c${cs}.log | tail -1 | grep -oE "[0-9]+")"
done
# c8 already ran (out_sens_171.log = 20445); copy for consistency
cp out_sens_171.log out_matrix_resc_c8.log 2>/dev/null
echo "resc cs=8 cycles=$(grep gpu_tot_sim_cycle out_matrix_resc_c8.log | tail -1 | grep -oE "[0-9]+") (from sens run)"
echo "RESC_C24_DONE"
