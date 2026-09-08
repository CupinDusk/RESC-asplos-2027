#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# LEGALITY-TAX axis (Phase 2): at EQUAL occupancy (TILE=0, all arms same CTA),
# sweep reuse to isolate the reuse RESC recovers that correct-ld.cg cannot.
# arms: cg=correct always-global(BACKEND=0, bypasses L1 per CUDA), resc(BACKEND=2),
#       ca=unsafe upper bound(BACKEND=4, L1-caches). All zero-SMEM => equal occupancy.
# Gap RESC/cg should GROW with reuse (=legality tax). ca~=RESC (unsafe ceiling).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
NC=${NC:-40}
W=legality_work; mkdir -p $W; rm -f $W/*
# build 3 arms (TILE=0, no compute tile needed for pure comm legality axis)
for arm in cg resc ca; do
  case $arm in
    cg)   be=0; extra="";;
    resc) be=2; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
    ca)   be=4; extra="";;
  esac
  nvcc -arch=sm_90 resc_e2e.cu -o $W/${arm}_l -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 $extra -lcudart 2>/dev/null
  patchelf --remove-rpath $W/${arm}_l 2>/dev/null
done
echo "# legality-tax sweep NC=$NC TILE=0 (equal occupancy) $(date -u +%H:%M:%S)"
echo "# payloadB N reuse arm cta cyc l1miss l2miss dram_rd"
# launch all configs in parallel
for pb in 512 2048 8192; do
  ae=$((pb/4))   # elems (payload bytes / 4)
  for cs in 2 8; do
    N=$([ $cs -eq 2 ] && echo 1 || echo 7)
    for ru in 1 2 4 8 16; do
      for arm in cg resc ca; do
        ( d=$W/r_${pb}_${cs}_${ru}_${arm}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config; cd $d
          timeout 300 ../${arm}_l cluster_size=$cs repeat=2 num_clusters=$NC avg_elems=$ae burst_period=16 reuse_passes=$ru > run.log 2>&1 ) &
      done
    done
  done
done
wait
# collect
for pb in 512 2048 8192; do
  for cs in 2 8; do
    N=$([ $cs -eq 2 ] && echo 1 || echo 7)
    for ru in 1 2 4 8 16; do
      for arm in cg resc ca; do
        raw=$(cat $W/r_${pb}_${cs}_${ru}_${arm}/run.log 2>/dev/null)
        cta=$(echo "$raw" | grep -oE "CTA/core = [0-9]+" | head -1 | grep -oE "[0-9]+$")
        cyc=$(echo "$raw" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
        l1m=$(echo "$raw" | grep L1D_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
        l2m=$(echo "$raw" | grep L2_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
        drd=$(echo "$raw" | grep gpgpu_n_mem_read_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
        echo "$pb $N $ru $arm ${cta:-NA} ${cyc:-NA} ${l1m:-NA} ${l2m:-NA} ${drd:-NA}"
      done
    done
  done
done > legality.dat
cat legality.dat
echo "LEGALITY_DONE"
