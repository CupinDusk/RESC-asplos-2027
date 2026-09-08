#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# ============================================================================
# RESC end-to-end master sweep -> single source-of-truth data file.
# Every paper table for the e2e section derives from this ONE file (roots out
# the table-drift bug: no hand-typed numbers).
# Sweeps the compute-tile occupancy staircase across all 4 arms.
#   arms: global(0, zero SMEM) DSM(1, +reserve) hybrid(3, +stage) RESC(2, zero SMEM)
# Emits per (arm, tile, cluster): occupancy CTA/core, limiter, residual L1 KB,
#   whole-kernel cycles, L1D miss-rate, L2 miss-rate, IPC, DRAM global r/w.
# ============================================================================
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/ptxas_shim:$PATH
cp cfg_h20.config gpgpusim.config

CI=${CI:-8}          # COMPUTE_ITERS (real FP compute passes)
NC=${NC:-40}         # num_clusters
RPT=${RPT:-8}        # repeat rounds
AE=${AE:-512}        # avg_elems (comm payload)
RU=${RU:-4}          # reuse_passes
OUT=${OUT:-e2e_master.dat}

# Build all 4 arms at each tile size (tile changes compile-time dyn_smem attr).
echo "# RESC e2e master sweep  CI=$CI NC=$NC RPT=$RPT AE=$AE RU=$RU  $(date -u +%Y-%m-%dT%H:%M:%SZ)" > $OUT
echo "# cols: arm tile_kb cluster N cta_core limiter l1_kb cycles l1d_miss_rate l2_miss_rate ipc dram_rd_g dram_wr_g" >> $OUT

for tile in 0 32 64 96 112 128; do
  for arm in global dsm hybrid resc; do
    case $arm in
      global) be=0; extra="";;
      dsm)    be=1; extra="";;
      hybrid) be=3; extra="-DSTAGE_ELEMS=8192";;   # competently-sized stage (32KB) per fair-hybrid
      resc)   be=2; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
    esac
    src=$([ $arm = hybrid ] && echo resc_hybrid_e2e.cu || echo resc_e2e.cu)
    bin=${arm}_m
    nvcc -arch=sm_90 $src -o $bin -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 \
         -DTILE_KB=$tile -DCOMPUTE_ITERS=$CI $extra -lcudart 2>/dev/null
    patchelf --remove-rpath $bin 2>/dev/null
    for cs in 2 8; do
      N=$([ $cs -eq 2 ] && echo 1 || echo 7)
      raw=$(timeout 400 ./$bin cluster_size=$cs repeat=$RPT num_clusters=$NC avg_elems=$AE burst_period=16 reuse_passes=$RU 2>/dev/null)
      cta=$(echo "$raw" | grep -oE "CTA/core = [0-9]+" | head -1 | grep -oE "[0-9]+")
      lim=$(echo "$raw" | grep -oE "limited by:.*" | head -1 | sed 's/limited by: *//' | tr ' ' '+' | sed 's/+$//')
      l1kb=$(echo "$raw" | grep -oE "Reconfigure L1 cache to [0-9]+KB" | head -1 | grep -oE "[0-9]+KB" | grep -oE "[0-9]+")
      cyc=$(echo "$raw" | grep "gpu_tot_sim_cycle" | tail -1 | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      l1m=$(echo "$raw" | grep "L1D_total_cache_miss_rate" | tail -1 | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      l2m=$(echo "$raw" | grep "L2_total_cache_miss_rate" | tail -1 | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      ipc=$(echo "$raw" | grep "gpu_tot_ipc" | tail -1 | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      drd=$(echo "$raw" | grep "gpgpu_n_mem_read_global" | tail -1 | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      dwr=$(echo "$raw" | grep "gpgpu_n_mem_write_global" | tail -1 | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      echo "$arm $tile $cs $N ${cta:-NA} ${lim:-NA} ${l1kb:-NA} ${cyc:-NA} ${l1m:-NA} ${l2m:-NA} ${ipc:-NA} ${drd:-NA} ${dwr:-NA}" | tee -a $OUT
    done
  done
done
echo "E2E_MASTER_DONE -> $OUT"
