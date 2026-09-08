#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# SELECTIVE-INVALIDATE ABLATION -- pipeline axis (pipe_chain), 60 MiB L2.
# Prereg: selective-invalidate-prereg.md S3 (fairness) / S6 (workloads).
#
# FAIRNESS RULE (prereg S3, non-negotiable): knobs 2 and 3 model RESC HARDWARE.
# Baselines that have no such hardware are NEVER run under 2 or 3.  Each baseline
# is run at BOTH of its legal options (0 = sync_restrict escape, 1 = plain
# cluster acquire) and is credited with whichever is FASTER -- the pairing LEAST
# favourable to RESC.
#
#   arm0 (ld.cg correct incumbent)  -> knobs 0,1        [no RESC hw]
#   arm1 (ld.ca unsafe overlay)     -> knobs 0,1        [no RESC hw]
#   arm2 (lesioned RESC, G=2)       -> knobs 0,1,2,3    [has RESC hw]
#   arm3 (full RESC, G=LAG+1, MW=1) -> knobs 0,1,2,3    [has RESC hw]
#
# CTRL reproduction (cfg_h20 4MiB, knob 0) is included for harness validity.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/pipe
mkdir -p $W
NC=4; NS=32; TL=4; TO=2700
BIN=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/pipe_chain

launch(){  # $1=cfgname $2=cfgfile $3=knob $4=LAG $5=arm
  local cfgname=$1 cfgfile=$2 inv=$3 LAG=$4 arm=$5 G MW d
  case $arm in
    arm0) G=1; MW=0;;
    arm1) G=0; MW=0;;
    arm2) G=2; MW=0;;
    arm3) G=$((LAG+1)); MW=1;;
  esac
  d=$W/${cfgname}_k${inv}_L${LAG}_${arm}; mkdir -p $d
  cp $cfgfile $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
  ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      $BIN $NC $NS $LAG $TL > run.log 2>&1; echo $? > exit.code ) &
}

# ---- CTRL reproduction: 4 MiB L2, knob 0, all four arms ----
for LAG in 1 2 3 4; do
  for arm in arm0 arm1 arm2 arm3; do
    launch CTRL cfg_h20.config 0 $LAG $arm
  done
done
wait
echo "PIPE_CTRL_DONE"

# ---- 60 MiB L2: baselines at knobs 0,1 only (fairness rule) ----
for LAG in 1 2 3 4; do
  for inv in 0 1; do
    for arm in arm0 arm1; do
      launch L60 cfg_h20_l2_60mb.config $inv $LAG $arm
    done
  done
done
wait
echo "PIPE_BASE_DONE"

# ---- 60 MiB L2: RESC arms at all four knobs ----
for LAG in 1 2 3 4; do
  for inv in 0 1 2 3; do
    for arm in arm2 arm3; do
      launch L60 cfg_h20_l2_60mb.config $inv $LAG $arm
    done
  done
done
wait
echo "PIPE_RESC_DONE"
echo "PIPE_ALL_DONE"
