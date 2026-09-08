#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_pool.config gpgpusim.config
./simxl_c8 512 > khist_full.log 2>&1
echo "KHIST_RUN_DONE exit=$?"
echo "=== KHIST lines ==="; grep "KHIST" khist_full.log | tail -3
echo "=== KHIST count ==="; grep -c "KHIST" khist_full.log
echo "=== ratio ==="; grep "ratio" khist_full.log | tail -1
