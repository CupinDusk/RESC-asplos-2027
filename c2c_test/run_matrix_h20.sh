#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_h20.config gpgpusim.config
mkdir -p matrix_h20
for arm in global dsm resc; do
  for cs in 2 4 8; do
    f=matrix_h20/${arm}_c${cs}.log
    [ -s "$f" ] && grep -q gpu_tot_sim_cycle "$f" && continue
    ./${arm}10240 cluster_size=$cs repeat=1 num_clusters=200 avg_elems=128 burst_period=16 reuse_passes=4 > "$f" 2>&1
    echo "$arm cs=$cs cyc=$(grep gpu_tot_sim_cycle "$f" | tail -1 | grep -oE "[0-9]+")"
  done
done
echo "MATRIX_H20_DONE"
