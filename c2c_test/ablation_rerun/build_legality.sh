#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Build the 4 legality-axis arms (resc_e2e), TILE_KB=0 equal occupancy.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=ablation_rerun/bin; mkdir -p $W
# arm0=cg(BACKEND0) arm1=ca(BACKEND4) arm2/arm3=resc(BACKEND2, differ only by env)
for arm in arm0 arm1 arm23; do
  case $arm in
    arm0)  be=0; extra="";;
    arm1)  be=4; extra="";;
    arm23) be=2; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
  esac
  nvcc -arch=sm_90 resc_e2e.cu -o $W/${arm}_l -DBACKEND=$be -DRESERVE_ELEMS=8192 \
       -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 $extra -lcudart 2>&1 | tail -3
  patchelf --remove-rpath $W/${arm}_l 2>/dev/null
  echo "built $arm -> $(md5sum $W/${arm}_l | cut -c1-12)"
done
