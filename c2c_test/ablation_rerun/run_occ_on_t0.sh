#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# ON60 TILE=0 occupancy points, run standalone so they are not gated behind the
# slow TILE=96 dsm arm. Matched parameters to occS OFF60_t0_* (CI=2,NC=8,RPT=2).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=ablation_rerun/occT; mkdir -p $W
B=ablation_rerun/occSbin
for arm in global dsm arm2 arm3; do
  case $arm in
    global) bin=global_t0; G=1; MW=0;;
    dsm)    bin=dsm_t0;    G=1; MW=0;;
    arm2)   bin=resc_t0;   G=2; MW=0;;
    arm3)   bin=resc_t0;   G=3; MW=1;;
  esac
  d=$W/ON60_t0_${arm}; mkdir -p $d
  cp cfg_h20_l2_60mb.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 1\n" >> $d/gpgpusim.config
  ( cd $d && timeout 2400 env RESC_GEN_RING=$G RESC_MW=$MW ../../../$B/$bin \
      cluster_size=8 repeat=2 num_clusters=8 avg_elems=512 burst_period=16 reuse_passes=4 \
      > run.log 2>&1; echo $? > exit.code ) &
done
wait
echo OCCT_DONE
