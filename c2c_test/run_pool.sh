#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
mkdir -p pool
cp cfg_pool.config gpgpusim.config
echo "=== SHARED-POOL (collapse=2) held-out predict START $(date) ==="
for cs in 2 8; do
  for nf in 64 128 256 384 512 768 1024; do
    ./simxl_c$cs $nf > pool/c${cs}_nf${nf}.log 2>&1
    echo "c$cs nf=$nf : $(grep -E 'ratio' pool/c${cs}_nf${nf}.log | tail -1)"
  done
done
echo "POOL_PREDICT_DONE"
