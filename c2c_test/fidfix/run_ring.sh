#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# D1 HEADLINE TEST: does the ring's off-chip-traffic win survive a 60 MiB L2?
# Pre-registered prediction (sim-fidelity-fix-prereg.md 2b): "SHRINKS OR VANISHES --
# the 33 MB working set that exceeded a 4 MiB L2 now fits in 60 MiB; the capacity
# regime the win lived in DOES NOT EXIST on the target part."
# Same benchmark, same args as the published result; arms differ ONLY in ring depth.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
BASE=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/ring
mkdir -p $BASE
# cfg: l2_4mb (reproduce published) and l2_60mb (the fix). invalidate OFF in both so
# the ONLY variable is L2 capacity (D1 isolated from D2).
for cfg in l2_4mb l2_60mb; do
  case $cfg in
    l2_4mb)  SRC=cfg_h20.config ;;
    l2_60mb) SRC=cfg_h20_l2_60mb.config ;;
  esac
  for G in 2 3; do
    d=$BASE/${cfg}_ring${G}
    mkdir -p $d && cd $d
    cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/$SRC gpgpusim.config
    printf "\n-gpgpu_cluster_barrier_invalidate_l1 0\n" >> gpgpusim.config
    cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/pipe_chain .
    RESC_GEN_RING=$G ./pipe_chain 64 32 2 16 > run.log 2>&1
    echo "$cfg G=$G : $(grep -E 'PIPE_CHAIN' run.log | tail -1)"
    cd $BASE
  done
done
echo "RING_ALL_DONE"
