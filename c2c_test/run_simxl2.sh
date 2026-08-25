#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_cal195.config gpgpusim.config
# push c8 to larger payloads where silicon crossed (>1KB), + confirm the flat trend
for nf in 2048 4096; do
  ./simxl_c8 $nf > simxl/out_c8_nf${nf}.log 2>&1
  echo "c8 nf=$nf : $(grep -E 'ratio' simxl/out_c8_nf${nf}.log | tail -1)"
done
echo "SIMXL2_DONE"
