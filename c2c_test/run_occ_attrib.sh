#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# OCCUPANCY attribution: does RESC's coherence design point (arm3, G>=3) change
# occupancy or cycles vs a safe single keep-valid interval (arm2, G=2)? Both are
# the SAME zero-SMEM binary; the gen-ring env cannot change launch-time SMEM
# reservation, so occupancy should be IDENTICAL => occupancy is PLACEMENT.
# Incumbent that pays the DSM occupancy tax = dsm_m (reserves SMEM). global_m =
# correct global (zero SMEM but no residency). All real runs, cfg_h20.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
OUT=ATTRIBUTION_OCCUPANCY.dat
echo "# OCCUPANCY attribution (arm2 G=2 vs arm3 G=3, same binary) $(date -u +%Y-%m-%dT%H:%M:%SZ)" > $OUT
echo "# resc_m rebuilt per TILE would be needed; using existing resc_m built at TILE=128." >> $OUT
echo "# arm tile cluster G cta_core cycles l1miss note" >> $OUT
# resc_m as-built (TILE compiled in). We rerun the SAME binary under G=2 and G=3.
for cs in 2 8; do
  for G in 2 3; do
    raw=$(timeout 400 env RESC_GEN_RING=$G ./resc_m cluster_size=$cs repeat=8 num_clusters=40 avg_elems=512 burst_period=16 reuse_passes=4 2>/dev/null)
    cta=$(echo "$raw"|grep -oE "CTA/core = [0-9]+"|head -1|grep -oE "[0-9]+$")
    cyc=$(echo "$raw"|grep "gpu_tot_sim_cycle"|tail -1|sed -E "s/.*= *//"|grep -oE "[0-9]+"|head -1)
    l1m=$(echo "$raw"|grep "L1D_total_cache_miss_rate"|tail -1|sed -E "s/.*= *//"|grep -oE "[0-9.]+"|head -1)
    echo "resc cs=$cs G=$G cta=${cta:-NA} cyc=${cyc:-NA} l1miss=${l1m:-NA}" | tee -a $OUT
  done
  # incumbent DSM (pays SMEM tax) and correct-global for the same point
  for arm in dsm global; do
    bin=${arm}_m
    raw=$(timeout 400 ./$bin cluster_size=$cs repeat=8 num_clusters=40 avg_elems=512 burst_period=16 reuse_passes=4 2>/dev/null)
    cta=$(echo "$raw"|grep -oE "CTA/core = [0-9]+"|head -1|grep -oE "[0-9]+$")
    cyc=$(echo "$raw"|grep "gpu_tot_sim_cycle"|tail -1|sed -E "s/.*= *//"|grep -oE "[0-9]+"|head -1)
    l1m=$(echo "$raw"|grep "L1D_total_cache_miss_rate"|tail -1|sed -E "s/.*= *//"|grep -oE "[0-9.]+"|head -1)
    echo "$arm cs=$cs G=- cta=${cta:-NA} cyc=${cyc:-NA} l1miss=${l1m:-NA}" | tee -a $OUT
  done
done
echo "OCC_ATTRIB_DONE" | tee -a $OUT
