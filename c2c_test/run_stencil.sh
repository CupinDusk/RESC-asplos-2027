#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
NC=${NC:-4}; IT=${IT:-16}
for arm in cg dsm resc noncoh; do
  d=str_${arm}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
  ( cd $d; timeout 600 ../st_$arm num_clusters=$NC iters=$IT > o.log 2>&1; echo "$arm EXIT=$?" > done.flag ) &
done
wait
echo STENCIL_ALL_DONE
