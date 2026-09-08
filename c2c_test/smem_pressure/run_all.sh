#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=smem_pressure
run(){ d=$W/r_$1; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config; (cd $d; env $2 ../$3 cluster_size=8 repeat=1 num_clusters=8 avg_elems=512 burst_period=16 > run.log 2>&1); echo "$1 done exit=$?"; }
run arm0_cg     ""             cg_sp
run DSM         ""             dsm_sp
run RESC_fwd    ""             resc_sp
run faithfulHLRC RESC_NO_FWD=1 resc_sp
echo ALL_DONE
