#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_h20.config gpgpusim.config
for nf in 16 64 256 1024; do
  ./simxl_c8 $nf > h20val/v2_c8_nf${nf}.log 2>&1
  echo "c8 nf=$nf : $(grep 'nfloats=' h20val/v2_c8_nf${nf}.log | tail -1)"
done
echo "H20VAL2_DONE"
