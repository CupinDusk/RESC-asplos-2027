#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Parallel e2e sweep: build all arm/tile binaries, launch all runs concurrently
# (192 cores, each sim single-threaded), collect into single source-of-truth file.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
cp cfg_h20.config gpgpusim.config

CI=${CI:-8}; CE=${CE:-2048}; NC=${NC:-96}; RPT=${RPT:-1}; AE=${AE:-512}; RU=${RU:-4}
TILES=${TILES:-"0 32 64 96 128"}
CLUSTERS=${CLUSTERS:-"2 8"}
WORK=e2e_par_work; mkdir -p $WORK; rm -f $WORK/*.out

echo "# building binaries $(date -u +%H:%M:%S)"
for tile in $TILES; do
  for arm in global dsm hybrid resc; do
    case $arm in
      global) be=0; src=resc_e2e.cu; extra="";;
      dsm) be=1; src=resc_e2e.cu; extra="";;
      hybrid) be=3; src=resc_hybrid_e2e.cu; extra="-DSTAGE_ELEMS=8192";;
      resc) be=2; src=resc_e2e.cu; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
    esac
    nvcc -arch=sm_90 $src -o $WORK/${arm}_t${tile} -DBACKEND=$be -DRESERVE_ELEMS=8192 \
         -DSTRIDE_ELEMS=16384 -DTILE_KB=$tile -DCOMPUTE_ITERS=$CI -DCOMPUTE_ELEMS=$CE $extra -lcudart 2>/dev/null
    patchelf --remove-rpath $WORK/${arm}_t${tile} 2>/dev/null
  done
done
echo "# launching all runs in parallel $(date -u +%H:%M:%S)"
# Each run needs its own gpgpusim.config in its own cwd -> use a per-run subdir
for tile in $TILES; do
  for arm in global dsm hybrid resc; do
    for cs in $CLUSTERS; do
      (
        d=$WORK/run_${arm}_t${tile}_c${cs}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
        cd $d
        ../${arm}_t${tile} cluster_size=$cs repeat=$RPT num_clusters=$NC avg_elems=$AE burst_period=16 reuse_passes=$RU > run.log 2>&1
      ) &
    done
  done
done
wait
echo "# collecting $(date -u +%H:%M:%S)"
OUT=${OUT:-e2e_par.dat}
echo "# RESC e2e parallel sweep CI=$CI NC=$NC RPT=$RPT AE=$AE RU=$RU $(date -u +%Y-%m-%dT%H:%M:%SZ)" > $OUT
echo "# arm tile cluster N cta limiter l1kb cyc l1miss l2miss ipc drd dwr" >> $OUT
for tile in $TILES; do
  for arm in global dsm hybrid resc; do
    for cs in $CLUSTERS; do
      N=$([ $cs -eq 2 ] && echo 1; [ $cs -eq 4 ] && echo 3; [ $cs -eq 8 ] && echo 7)
      raw=$(cat $WORK/run_${arm}_t${tile}_c${cs}/run.log 2>/dev/null)
      cta=$(echo "$raw" | grep -oE "CTA/core = [0-9]+" | head -1 | grep -oE "[0-9]+$")
      lim=$(echo "$raw" | grep -oE "limited by:.*" | head -1 | sed "s/limited by: *//" | tr " " "+" | sed "s/+$//")
      l1kb=$(echo "$raw" | grep -oE "Reconfigure L1 cache to [0-9]+KB" | head -1 | grep -oE "[0-9]+KB" | grep -oE "[0-9]+")
      cyc=$(echo "$raw" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      l1m=$(echo "$raw" | grep L1D_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      l2m=$(echo "$raw" | grep L2_total_cache_miss_rate | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      ipc=$(echo "$raw" | grep gpu_tot_ipc | sed -E "s/.*= *//" | grep -oE "[0-9.]+" | head -1)
      drd=$(echo "$raw" | grep gpgpu_n_mem_read_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      dwr=$(echo "$raw" | grep gpgpu_n_mem_write_global | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
      echo "$arm $tile $cs $N ${cta:-NA} ${lim:-NA} ${l1kb:-NA} ${cyc:-NA} ${l1m:-NA} ${l2m:-NA} ${ipc:-NA} ${drd:-NA} ${dwr:-NA}" >> $OUT
    done
  done
done
cat $OUT
echo "E2E_PAR_DONE -> $OUT"
