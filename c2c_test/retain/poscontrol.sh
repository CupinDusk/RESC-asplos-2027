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
D=$T/retain/poscontrol_R2_cl32_pv256
mkdir -p $D
cp $T/cfg_h20_tinyL1.config $D/gpgpusim.config
printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> $D/gpgpusim.config
cd $D
timeout 7200 env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 RESC_RETAIN=1 RESC_RETAIN_Q=100 \
  $T/retain/bin/rt_arm2 2 32 2 32 256 > run.log 2>&1
echo $? > exit.code
echo "POSCTL_EXIT=$(cat exit.code)"
