#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# DETECTION-FLOOR CALIBRATION sweep.
# Dependent-read (pointer-chase) microbench, two arms differing ONLY in cache
# operator (.ca = L1-allocating vs .cg = L1-bypassing).  Sweep warps/SM to find
# where an L1-vs-L2 difference is still detectable.  Establishes the sensitivity
# of the instrument, so a null can be distinguished from a blind instrument.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
BASE=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/detfloor
mkdir -p $BASE
for W in 1 2 4 8 16 32; do
  for A in 0 1; do
    b=$BASE/d_w${W}_a${A}
    nvcc -arch=sm_90 fidfix/detfloor.cu -o $b -DWARPS=$W -DARM=$A -DCHASE_KB=16 -DSTEPS=512 -DNBLOCKS=1 -lcudart > $BASE/build_w${W}_a${A}.log 2>&1
    patchelf --remove-rpath $b
  done
done
echo "DETFLOOR_BUILD_DONE"
for W in 1 2 4 8 16 32; do
  for A in 0 1; do
    d=$BASE/r_w${W}_a${A}
    mkdir -p $d && cd $d
    cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/cfg_h20_l2_60mb.config gpgpusim.config
    printf "\n-gpgpu_cluster_barrier_invalidate_l1 1\n" >> gpgpusim.config
    cp $BASE/d_w${W}_a${A} .
    ./d_w${W}_a${A} > run.log 2>&1
    echo "$(grep DETFLOOR run.log | tail -1)"
    cd $BASE
  done
done
echo "DETFLOOR_ALL_DONE"
