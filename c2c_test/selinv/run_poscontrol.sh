#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# SELECTIVITY POSITIVE CONTROL (prereg S5).
# Build selprobe once, run at knobs 0,1,2,3.  The required signature is that
# knob 2 preserves the UNMARKED tile (like knob 0) while STILL invalidating the
# MARKED buffer (like knob 1).  If 2 == 0 or 2 == 1 on both metrics, the
# implementation is wrong and the ablation must not be run.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/pc
mkdir -p $W

nvcc -arch=sm_90 selprobe.cu -o $W/selprobe \
     -DEPOCHS=32 -DTILE_KB=16 -DCOMM_KB=1 -DCS=2 -DNCLUSTERS=2 \
     -lcudart > $W/build.log 2>&1 || { echo BUILD_FAIL; tail -30 $W/build.log; exit 1; }
patchelf --remove-rpath $W/selprobe
echo "BUILD_OK"

for inv in 0 1 2 3; do
  d=$W/k$inv; mkdir -p $d
  cp cfg_h20.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" "$inv" >> $d/gpgpusim.config
  cp $W/selprobe $d/
  ( cd $d && timeout 1800 env RESC_QUIET=1 ./selprobe > run.log 2>&1; echo $? > exit.code ) &
done
wait
echo "PC_DONE"
