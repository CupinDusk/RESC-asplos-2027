#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
arm=$1; bin=$2; RT=$3
D=$T/maxunion/smoke_${arm}
mkdir -p $D; cp $T/cfg_h20_smem228.config $D/gpgpusim.config
printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> $D/gpgpusim.config
cd $D
# ntasks=4 (smoke), big_every=2, comm_lines=64, priv_lines=256
timeout 5400 env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 ${RT:+RESC_RETAIN=1 RESC_RETAIN_Q=100} \
  $T/maxunion/bin/$bin 2 4 2 64 256 > run.log 2>&1
echo $? > exit.code
