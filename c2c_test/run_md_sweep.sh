#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Full MD experiment: 3 arms x pbar sweep, whole-kernel cycles + correctness + traffic.
# Reports RESC/cg (legality axis) and RESC/dsm (occupancy axis) whole-kernel, plus
# dram traffic (for shared-written-fraction f) and checksum (correctness).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
NC=${NC:-16}; RPT=${RPT:-2}
echo "# MD sweep NC=$NC RPT=$RPT  arm pbar cyc checksum l1miss dram_rd dram_wr"
for pbar in 16 32 48; do
  for arm in cg dsm resc; do
    d=mdsw_${arm}_${pbar}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
    ( cd $d
      raw=$(timeout 600 ../md_$arm num_clusters=$NC repeat=$RPT pbar=$pbar 2>/dev/null | grep -vE "\[RESC\]")
      cyc=$(echo "$raw" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      cs=$(echo "$raw" | grep OUT_CHECKSUM | grep -oE "sum=[0-9.]+")
      l1=$(echo "$raw" | grep L1D_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      dr=$(echo "$raw" | grep gpgpu_n_mem_read_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      dw=$(echo "$raw" | grep gpgpu_n_mem_write_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      echo "$arm $pbar ${cyc:-NA} ${cs:-NA} ${l1:-NA} ${dr:-NA} ${dw:-NA}" ) &
  done
  wait
done > md_sweep.dat
cat md_sweep.dat
echo MD_SWEEP_DONE
