#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Exp3: co-demo occupancy + legality on ONE real kernel (stencil halo exchange).
# TILE_DIM 64/96 make DSMs map_shared_rank staging buffer cost resident CTAs
# (occupancy axis) while RESC keeps halo L1-resident across iters (legality axis).
# Binaries renamed k_<arm>_td<NN> to avoid the sim sm_NN name-parse bug.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
W=exp3_occ; NC=${NC:-8}; IT=${IT:-16}
echo "# Exp3 stencil co-demo NC=$NC iters=$IT cluster=4"
echo "# tile arm cyc cta checksum"
for td in 32 64 96; do
  for arm in cg dsm resc noncoh; do
    d=$W/run_${arm}_${td}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
    ( cd $d; timeout 600 ../k_${arm}_td${td} num_clusters=$NC iters=$IT > run.log 2>&1 ) &
  done
  wait
done
for td in 32 64 96; do
  for arm in cg dsm resc noncoh; do
    d=$W/run_${arm}_${td}
    cyc=$(grep gpu_tot_sim_cycle $d/run.log | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
    cta=$(grep -oE "CTA/core = [0-9]+" $d/run.log | head -1 | grep -oE "[0-9]+$")
    ck=$(grep OUT_CHECKSUM $d/run.log | grep -oE "sum=[-0-9.]+" | head -1)
    echo "$td $arm ${cyc:-NA} ${cta:-NA} ${ck:-NA}"
  done
done
echo EXP3_DONE
