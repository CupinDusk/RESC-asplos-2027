#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export PATH=/usr/local/cuda/bin:$PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt
mkdir -p build && cd build
echo "===== C2C BUILD START $(date) ====="
cmake -DCMAKE_BUILD_TYPE=Release .. > cmake_c2c.log 2>&1
echo "CMAKE_EXIT=$?"
make -j6 > make_c2c.log 2>&1
echo "MAKE_EXIT=$?"
tail -3 make_c2c.log
ls -la lib/libcudart.so* 2>/dev/null
echo "===== C2C BUILD DONE $(date) ====="
