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
# $1 = tag, $2 = cfg, $3.. = env assignments (as VAR=VAL), then -- args
tag=$1; cfg=$2; shift 2
D=$T/retain/diag_$tag
mkdir -p $D; cp $T/$cfg $D/gpgpusim.config
printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> $D/gpgpusim.config
cd $D
timeout 3600 env RESC_QUIET=1 "$@" $T/retain/bin/rt_arm2 2 32 2 32 256 > run.log 2>&1
echo $? > exit.code
