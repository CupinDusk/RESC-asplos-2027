#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
# fill pbar 32,48 with 900s budget; write per-run o.log so checksum is retrievable
for pbar in 32 48; do
  for arm in cg dsm resc; do
    d=mdf_${arm}_${pbar}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
    ( cd $d; timeout 900 ../md_$arm num_clusters=16 repeat=2 pbar=$pbar 2>/dev/null | grep -vE "\[RESC\]" > o.log ) &
  done
done
wait
echo "arm pbar cyc checksum l1miss dram_rd"
for pbar in 16 32 48; do
  for arm in cg dsm resc; do
    dd=mdf_${arm}_${pbar}; [ $pbar -eq 16 ] && dd=mdsw_${arm}_16
    f=$dd/o.log
    cyc=$(grep gpu_tot_sim_cycle $f 2>/dev/null | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
    cs=$(grep OUT_CHECKSUM $f 2>/dev/null | grep -oE "sum=[-0-9.]+")
    l1=$(grep L1D_total_cache_miss_rate $f 2>/dev/null | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
    dr=$(grep gpgpu_n_mem_read_global $f 2>/dev/null | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
    echo "$arm $pbar ${cyc:-NA} ${cs:-NA} ${l1:-NA} ${dr:-NA}"
  done
done > md_full.dat
cat md_full.dat
echo MD_FILL_DONE
