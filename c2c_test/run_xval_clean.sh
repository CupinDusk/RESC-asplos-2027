#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_h20.config gpgpusim.config
mkdir -p xval_clean
# simxl_c* = simxover_light (nclusters=8, completes in sim). ONLY cfg_h20 (78 SM, H20 clocks).
for cs in 2 4 8; do
  for nf in 64 256 512 1024; do
    ./simxl_c$cs $nf > xval_clean/c${cs}_nf${nf}.log 2>&1
    r=$(grep "nfloats=" xval_clean/c${cs}_nf${nf}.log | tail -1)
    echo "c${cs} nf${nf}: ${r}"
  done
done
echo "XVAL_CLEAN_DONE"
