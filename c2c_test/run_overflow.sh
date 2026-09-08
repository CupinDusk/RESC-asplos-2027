#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
# Build ALL FOUR arms at SAME RESERVE_ELEMS=8192 (fair hotset budget) + STRIDE=16384.
# hybrid additionally gets STAGE_ELEMS=512 (small 2KB SMEM stage) so hotset(up to 8192)>>stage.
for arm in global dsm resc; do
  be=$([ $arm = global ] && echo 0; [ $arm = dsm ] && echo 1; [ $arm = resc ] && echo 2)
  rm=$([ $arm = resc ] && echo "-DREAD_MODE=1 -DWRITE_MODE=1" || echo "")
  nvcc -arch=sm_90 resc.cu -o ${arm}_of -DBACKEND=$be -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 $rm -lcudart 2>/dev/null && patchelf --remove-rpath ${arm}_of 2>/dev/null
done
nvcc -arch=sm_90 resc_hybrid.cu -o hybrid_of -DBACKEND=3 -DRESERVE_ELEMS=8192 -DSTAGE_ELEMS=512 -DSTRIDE_ELEMS=16384 -lcudart 2>/dev/null && patchelf --remove-rpath hybrid_of 2>/dev/null
echo "built: $(ls -1 *_of 2>/dev/null | tr "\n" " ")"
echo "arm payloadElems N cyc"
for ae in 512 2048 8192; do   # 512=fits stage; 2048,8192 = exceed 512-elem stage -> overflow regime
  for cs in 2 8; do
    N=$([ $cs -eq 2 ] && echo 1 || echo 7)
    for arm in global dsm resc hybrid; do
      c=$(timeout 300 ./${arm}_of cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$ae burst_period=16 reuse_passes=4 2>/dev/null | grep gpu_tot_sim_cycle | tail -1 | grep -oE "[0-9]+")
      echo "$arm $ae $N $c"
    done
  done
done
echo "OVERFLOW_DONE"
