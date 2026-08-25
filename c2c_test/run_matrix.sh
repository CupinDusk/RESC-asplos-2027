#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_171.config gpgpusim.config
for arm in global dsm; do
  for cs in 2 4 8; do
    echo "===== $arm cs=$cs START $(date) ====="
    ./${arm}10240 cluster_size=$cs repeat=1 num_clusters=200 avg_elems=128 burst_period=16 reuse_passes=4 > out_matrix_${arm}_c${cs}.log 2>&1
    echo "$arm cs=$cs EXIT=$? cycles=$(grep gpu_tot_sim_cycle out_matrix_${arm}_c${cs}.log | tail -1 | grep -oE "[0-9]+")"
  done
done
echo "MATRIX_DONE"
