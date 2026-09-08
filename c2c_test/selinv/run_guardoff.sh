#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# GUARD-OFF EQUIVALENCE PROOF (selective-invalidate change).
# Prereg: selective-invalidate-prereg.md SHA-256 8ebabe19... frozen 2026-09-03T11:11:43Z
# Requirement (prereg S4): at knob 0 and knob 1 the NEW build must reproduce the
# PREVIOUS build (md5 c554ded8...) bit-exactly.  Three independent checks:
#   G1 barrier probe, 4 arms x 2 knobs, vs pinned values in
#      sim-fidelity-fix-methodology.md S4
#   G2 ablation sample pipe_chain 4 32 2 4, 4 arms, CTRL(0) and ON60(1),
#      vs ablation-rerun-methodology.md S3.1
#   G3 sanity gate (already run and PASSED: 2656/2868/0.926)
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv
mkdir -p $W/g1 $W/g2

echo "=== G1: barrier probe, reuse the ALREADY-BUILT binaries from fidfix/probe"
PB=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/probe
for inv in 0 1; do
  for arm in A B C D; do
    d=$W/g1/r_${arm}_inv${inv}; mkdir -p $d
    cp cfg_h20.config $d/gpgpusim.config
    printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" "$inv" >> $d/gpgpusim.config
    cp $PB/p_$arm $d/
    ( cd $d && timeout 1200 ./p_$arm > run.log 2>&1; echo $? > exit.code ) &
  done
done
wait
echo "G1_DONE"

echo "=== G2: ablation sample pipe_chain 4 32 2 4, 4 arms x {CTRL(0), ON60(1)}"
for spec in CTRL:cfg_h20.config:0 ON60:cfg_h20_l2_60mb.config:1; do
  cfgname=${spec%%:*}; rest=${spec#*:}; cfgfile=${rest%%:*}; inv=${rest##*:}
  for arm in arm0 arm1 arm2 arm3; do
    case $arm in
      arm0) G=1; MW=0;;
      arm1) G=0; MW=0;;
      arm2) G=2; MW=0;;
      arm3) G=3; MW=1;;   # LAG=2 -> G=LAG+1=3
    esac
    d=$W/g2/${cfgname}_L2_${arm}; mkdir -p $d
    cp $cfgfile $d/gpgpusim.config
    printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" "$inv" >> $d/gpgpusim.config
    ( cd $d && timeout 2400 env RESC_GEN_RING=$G RESC_MW=$MW \
        /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/pipe_chain 4 32 2 4 \
        > run.log 2>&1; echo $? > exit.code ) &
  done
done
wait
echo "G2_DONE"
echo "GUARDOFF_ALL_DONE"
