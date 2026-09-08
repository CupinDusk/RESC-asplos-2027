#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
cp cfg_h20.config gpgpusim.config
TILE=${TILE:-96}; CI=${CI:-4}; NC=${NC:-4}; RPT=${RPT:-2}
echo "# operating point: TILE=$TILE CI=$CI NC=$NC RPT=$RPT cluster=8 (N=7)"
echo "arm cta cyc l1miss dram_rd dram_wr"
for arm in global dsm hybrid resc; do
  case $arm in
    global) be=0; src=resc_e2e.cu; extra="";;
    dsm) be=1; src=resc_e2e.cu; extra="";;
    hybrid) be=3; src=resc_hybrid_e2e.cu; extra="-DSTAGE_ELEMS=8192";;
    resc) be=2; src=resc_e2e.cu; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
  esac
  nvcc -arch=sm_90 $src -o ${arm}_op -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=$TILE -DCOMPUTE_ITERS=$CI $extra -lcudart 2>/dev/null
  patchelf --remove-rpath ${arm}_op 2>/dev/null
  raw=$(timeout 300 ./${arm}_op cluster_size=8 repeat=$RPT num_clusters=$NC avg_elems=512 burst_period=16 reuse_passes=4 2>/dev/null)
  cta=$(echo "$raw" | grep -oE "CTA/core = [0-9]+" | head -1 | grep -oE "[0-9]+$")
  cyc=$(echo "$raw" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  l1m=$(echo "$raw" | grep L1D_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
  drd=$(echo "$raw" | grep gpgpu_n_mem_read_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  dwr=$(echo "$raw" | grep gpgpu_n_mem_write_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  echo "$arm ${cta:-NA} ${cyc:-NA} ${l1m:-NA} ${drd:-NA} ${dwr:-NA}"
done
echo OP_DONE
