#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/build
echo "===== COLLAPSE BUILD START $(date) ====="
make -j6 2>&1 | tail -20
echo "MAKE_EXIT=${PIPESTATUS[0]}"
ls -la lib/libcudart.so.12 2>/dev/null
echo "===== COLLAPSE BUILD DONE $(date) ====="
