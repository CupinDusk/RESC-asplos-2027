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
D=$T/retain/gate_post
mkdir -p $D
cp $T/cfg_cal195.config $D/gpgpusim.config
cp $T/simxl_c8 $D/ 2>/dev/null
cd $D
# KNOB OFF: RESC_RETAIN deliberately UNSET
env -u RESC_RETAIN -u RESC_RETAIN_Q -u RESC_RETAIN_W ./simxl_c8 256 > gate.log 2>&1
echo "GATE_EXIT=$?"
