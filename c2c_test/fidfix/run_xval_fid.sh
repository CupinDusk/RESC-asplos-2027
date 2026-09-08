#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# KILL-V RE-VALIDATION: the exact instrument that produced validation_crossover.dat
# (run_xval_clean.sh), re-run under each fidelity configuration so the error vs real
# H20 can be recomputed identically.
#   cfg=base   : pre-fix model, invalidate OFF  (must reproduce the published table)
#   cfg=inv    : 4 MiB L2, cluster-acquire invalidate ON      (D2 alone)
#   cfg=l2     : 60 MiB L2, invalidate OFF                    (D1 alone)
#   cfg=both   : 60 MiB L2, invalidate ON                     (D1 + D2)
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
BASE=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/xval
mkdir -p $BASE

for cfg in base inv l2 both; do
  case $cfg in
    base) SRC=cfg_h20.config;         INV=0 ;;
    inv)  SRC=cfg_h20.config;         INV=1 ;;
    l2)   SRC=cfg_h20_l2_60mb.config; INV=0 ;;
    both) SRC=cfg_h20_l2_60mb.config; INV=1 ;;
  esac
  for cs in 2 4 8; do
    for nf in 64 256 512 1024; do
      d=$BASE/${cfg}_c${cs}_nf${nf}
      mkdir -p $d && cd $d
      cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/$SRC gpgpusim.config
      printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" "$INV" >> gpgpusim.config
      cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/simxl_c$cs .
      ./simxl_c$cs $nf > run.log 2>&1
      echo "$cfg c$cs nf$nf : $(grep 'nfloats=' run.log | tail -1)"
      cd $BASE
    done
  done
done
echo "XVAL_FID_DONE"
