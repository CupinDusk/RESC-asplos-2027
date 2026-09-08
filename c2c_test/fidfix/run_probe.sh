#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Barrier-probe divergence test: arms A (1 barrier) and B (32 barriers).
# Pre-fix both give 512 GLOBAL_ACC_R misses. With the acquire fix ON they MUST diverge.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
BASE=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/probe
mkdir -p $BASE
# Build the two arms (same source as the original j1 probe)
for arm in A B C D; do
  case $arm in
    A) EP=1;  PS=32; SM=0 ;;
    B) EP=32; PS=1;  SM=0 ;;
    C) EP=32; PS=1;  SM=1 ;;   # positive control: membar.gl + barrier
    D) EP=1;  PS=32; SM=2 ;;   # negative control: no sync
  esac
  nvcc -arch=sm_90 j1_gate/j1_probe.cu -o $BASE/p_$arm \
       -DEPOCHS=$EP -DPASSES=$PS -DSYNCMODE=$SM -DTILE_KB=16 -DCS=2 -DNCLUSTERS=2 \
       -lcudart > $BASE/build_$arm.log 2>&1
  patchelf --remove-rpath $BASE/p_$arm
done
echo "BUILD_DONE"
# Run each arm under invalidate ON and OFF
for inv in 1 0; do
  for arm in A B C D; do
    d=$BASE/r_${arm}_inv${inv}
    mkdir -p $d && cd $d
    cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/cfg_h20.config gpgpusim.config
    printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" "$inv" >> gpgpusim.config
    cp $BASE/p_$arm .
    ./p_$arm > run.log 2>&1
    echo "arm=$arm inv=$inv done"
    cd $BASE
  done
done
echo "PROBE_ALL_DONE"
