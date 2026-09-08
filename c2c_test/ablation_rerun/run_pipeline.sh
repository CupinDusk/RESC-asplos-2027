#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Ablation re-run: pipeline axis (pipe_chain), 3 configs x 4 arms x LAG{1,2,3,4}
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=ablation_rerun/pipe; mkdir -p $W
NC=4; NS=32; TL=4; TO=1800
# cfgname:file:inv
for spec in CTRL:cfg_h20.config:0 OFF60:cfg_h20_l2_60mb.config:0 ON60:cfg_h20_l2_60mb.config:1; do
  cfgname=${spec%%:*}; rest=${spec#*:}; cfgfile=${rest%%:*}; inv=${rest##*:}
  for LAG in 1 2 3 4; do
    for arm in arm0 arm1 arm2 arm3; do
      case $arm in
        arm0) G=1; MW=0;;
        arm1) G=0; MW=0;;
        arm2) G=2; MW=0;;
        arm3) G=$((LAG+1)); MW=1;;
      esac
      d=$W/${cfgname}_L${LAG}_${arm}; mkdir -p $d
      cp $cfgfile $d/gpgpusim.config
      printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
      ( cd $d && timeout $TO env RESC_GEN_RING=$G RESC_MW=$MW ../../../pipe_chain $NC $NS $LAG $TL > run.log 2>&1; echo $? > exit.code ) &
    done
  done
  wait
done
echo PIPE_ALL_DONE
