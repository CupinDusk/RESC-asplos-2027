#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# SELECTIVE-INVALIDATE ABLATION -- legality axis (resc_e2e, TILE_KB=0), 60 MiB L2.
# Prereg S3 fairness rule: knobs 2/3 = RESC hardware, NEVER given to a baseline.
#   arm0 (ld.cg) -> knobs 0,1        arm1 (ld.ca) -> knobs 0,1
#   arm2/arm3 (RESC) -> knobs 0,1,2,3
# Reuses the EXACT binaries the previous ablation used (ablation_rerun/bin/*),
# so the columns are directly comparable.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/leg
B=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/ablation_rerun/bin
mkdir -p $W
NC=40; TO=2700

launch(){ # $1=cfgname $2=cfgfile $3=knob $4=payload $5=reuse $6=arm
  local cfgname=$1 cfgfile=$2 inv=$3 pb=$4 ru=$5 arm=$6 bin G MW d ae
  ae=$((pb/4))
  case $arm in
    arm0) bin=arm0_l;  G=1; MW=0;;
    arm1) bin=arm1_l;  G=0; MW=0;;
    arm2) bin=arm23_l; G=2; MW=0;;
    arm3) bin=arm23_l; G=3; MW=1;;
  esac
  d=$W/${cfgname}_k${inv}_p${pb}_r${ru}_${arm}; mkdir -p $d
  cp $cfgfile $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW $B/$bin \
      cluster_size=2 repeat=2 num_clusters=$NC avg_elems=$ae burst_period=16 reuse_passes=$ru \
      > run.log 2>&1; echo $? > exit.code ) &
}

# ---- CTRL reproduction: 4 MiB L2, knob 0 ----
for pb in 512 2048 8192; do
  for ru in 1 2 4 8 16; do
    for arm in arm0 arm1 arm2 arm3; do
      launch CTRL cfg_h20.config 0 $pb $ru $arm
    done
  done
  wait
done
echo "LEG_CTRL_DONE"

# ---- 60 MiB: baselines at knobs 0,1 ----
for pb in 512 2048 8192; do
  for ru in 1 2 4 8 16; do
    for inv in 0 1; do
      for arm in arm0 arm1; do
        launch L60 cfg_h20_l2_60mb.config $inv $pb $ru $arm
      done
    done
  done
  wait
done
echo "LEG_BASE_DONE"

# ---- 60 MiB: RESC arms at knobs 0,1,2,3 ----
for pb in 512 2048 8192; do
  for ru in 1 2 4 8 16; do
    for inv in 0 1 2 3; do
      for arm in arm2 arm3; do
        launch L60 cfg_h20_l2_60mb.config $inv $pb $ru $arm
      done
    done
  done
  wait
done
echo "LEG_RESC_DONE"
echo "LEG_ALL_DONE"
