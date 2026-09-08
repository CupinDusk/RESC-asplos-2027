#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
for arm in cg dsm; do
  d=exp3_occ/run_${arm}_64; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
  ( cd $d; ../k_${arm}_td64 num_clusters=8 iters=16 > run.log 2>&1; echo "${arm}64 EXIT=$?" > done.flag ) &
done
wait
echo SLOWARMS64_DONE
