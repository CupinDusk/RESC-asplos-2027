#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
for arm in cg dsm resc; do
  d=mds_$arm; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
  ( cd $d; timeout 500 ../md_$arm num_clusters=4 repeat=1 pbar=16 2>/dev/null | grep -vE "\[RESC\]" > o.log
    cs=$(grep OUT_CHECKSUM o.log | grep -oE "sum=[0-9.]+")
    cyc=$(grep gpu_tot_sim_cycle o.log | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
    echo "$arm $cs cyc=${cyc:-NONE}" )
done
echo MD_SMOKE_DONE
