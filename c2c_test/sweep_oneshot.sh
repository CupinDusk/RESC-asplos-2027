#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
mkdir -p sweep_oneshot
echo "arm ae cs cyc"
for ae in 8 32 128 512 2048; do
  for cs in 2 4 8; do
    for arm in global dsm resc; do
      f=sweep_oneshot/${arm}_ae${ae}_c${cs}.log
      [ -s "$f" ] && grep -q gpu_tot_sim_cycle "$f" && { echo "$arm $ae $cs cached"; continue; }
      timeout 400 ./${arm}10240 cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$ae burst_period=16 reuse_passes=1 > "$f" 2>&1
      c=$(grep gpu_tot_sim_cycle "$f" | tail -1 | grep -oE "[0-9]+")
      echo "$arm $ae $cs $c"
    done
  done
done
echo "ONESHOT_SWEEP_DONE"
