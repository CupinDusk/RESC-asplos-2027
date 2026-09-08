#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Parallel operating-point: large grid (forces multiple waves so occupancy matters)
# + bounded compute (COMPUTE_ELEMS) so 1-CTA arms finish. All 4 arms concurrent.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
TILE=${TILE:-96}; CI=${CI:-8}; CE=${CE:-2048}; NC=${NC:-96}; RPT=${RPT:-1}; CS=${CS:-8}
W=op_par_work; mkdir -p $W; rm -f $W/*
for arm in global dsm hybrid resc; do
  case $arm in
    global) be=0; src=resc_e2e.cu; extra="";;
    dsm) be=1; src=resc_e2e.cu; extra="";;
    hybrid) be=3; src=resc_hybrid_e2e.cu; extra="-DSTAGE_ELEMS=8192";;
    resc) be=2; src=resc_e2e.cu; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
  esac
  nvcc -arch=sm_90 $src -o $W/${arm}_op -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 \
       -DTILE_KB=$TILE -DCOMPUTE_ITERS=$CI -DCOMPUTE_ELEMS=$CE $extra -lcudart 2>/dev/null
  patchelf --remove-rpath $W/${arm}_op 2>/dev/null
done
echo "built $(date -u +%H:%M:%S); launching 4 arms parallel, TILE=$TILE CI=$CI CE=$CE NC=$NC CS=$CS RPT=$RPT"
for arm in global dsm hybrid resc; do
  ( d=$W/r_$arm; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config; cd $d
    ../${arm}_op cluster_size=$CS repeat=$RPT num_clusters=$NC avg_elems=512 burst_period=16 reuse_passes=4 > run.log 2>&1 ) &
done
wait
echo "arm cta cyc l1miss l2miss drd dwr"
for arm in global dsm hybrid resc; do
  raw=$(cat $W/r_$arm/run.log 2>/dev/null)
  cta=$(echo "$raw" | grep -oE "CTA/core = [0-9]+" | head -1 | grep -oE "[0-9]+$")
  cyc=$(echo "$raw" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  l1m=$(echo "$raw" | grep L1D_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
  l2m=$(echo "$raw" | grep L2_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
  drd=$(echo "$raw" | grep gpgpu_n_mem_read_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  dwr=$(echo "$raw" | grep gpgpu_n_mem_write_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  echo "$arm ${cta:-NA} ${cyc:-NA} ${l1m:-NA} ${l2m:-NA} ${drd:-NA} ${dwr:-NA}"
done
echo OP_PAR_DONE
