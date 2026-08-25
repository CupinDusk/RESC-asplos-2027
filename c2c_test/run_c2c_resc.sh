#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
echo "ldd: $(ldd ./resc10240 2>/dev/null | grep cudart)"
for cs in 2 4 8; do
  echo "===== C2C RESC cs=$cs START $(date) ====="
  ./resc10240 cluster_size=$cs repeat=1 num_clusters=200 avg_elems=128 burst_period=16 reuse_passes=4 > out_c2c_resc_c${cs}.log 2>&1
  echo "EXIT=$? cs=$cs cycles=$(grep gpu_tot_sim_cycle out_c2c_resc_c${cs}.log | tail -1 | grep -oE "[0-9]+") shares=$(grep -c "Read share SUCCESS" out_c2c_resc_c${cs}.log)"
done
echo "ALL_C2C_RESC_DONE"
