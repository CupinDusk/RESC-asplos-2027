#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_h20.config gpgpusim.config
mkdir -p h20val
echo "=== H20-config sim DSM-vs-global latency (small msg, latency-bound) START $(date) ==="
# small payloads to isolate latency (validation target: H20 real DSM small-msg wins)
for nf in 16 64 256; do
  ./simxl_c8 $nf > h20val/c8_nf${nf}.log 2>&1
  echo "c8 nf=$nf : $(grep -E 'ratio' h20val/c8_nf${nf}.log | tail -1)"
done
echo "H20VAL_DONE"
