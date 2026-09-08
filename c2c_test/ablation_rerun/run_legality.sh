#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Ablation re-run: legality axis (resc_e2e, TILE=0 equal occupancy).
# 3 configs x 4 arms x payload{512,2048,8192} x reuse{1,2,4,8,16}, cluster_size=2 (N=1).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=ablation_rerun/leg; mkdir -p $W
B=ablation_rerun/bin
NC=40; TO=1800
for spec in CTRL:cfg_h20.config:0 OFF60:cfg_h20_l2_60mb.config:0 ON60:cfg_h20_l2_60mb.config:1; do
  cfgname=${spec%%:*}; rest=${spec#*:}; cfgfile=${rest%%:*}; inv=${rest##*:}
  for pb in 512 2048 8192; do
    ae=$((pb/4))
    for ru in 1 2 4 8 16; do
      for arm in arm0 arm1 arm2 arm3; do
        case $arm in
          arm0) bin=arm0_l;  G=1; MW=0;;
          arm1) bin=arm1_l;  G=0; MW=0;;
          arm2) bin=arm23_l; G=2; MW=0;;
          arm3) bin=arm23_l; G=3; MW=1;;
        esac
        d=$W/${cfgname}_p${pb}_r${ru}_${arm}; mkdir -p $d
        cp $cfgfile $d/gpgpusim.config
        printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
        ( cd $d && timeout $TO env RESC_GEN_RING=$G RESC_MW=$MW ../../../$B/$bin \
            cluster_size=2 repeat=2 num_clusters=$NC avg_elems=$ae burst_period=16 reuse_passes=$ru \
            > run.log 2>&1; echo $? > exit.code ) &
      done
    done
  done
  wait
done
echo LEG_ALL_DONE
