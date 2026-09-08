#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# REDUCED-SCALE occupancy axis so the cycle columns actually complete.
# Same arms/configs; smaller repeat/compute/clusters. Absolute cycles are NOT
# comparable to the published staircase; the ON-vs-OFF comparison at matched
# parameters IS valid, which is the only thing this axis needs to answer.
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=ablation_rerun/occS; mkdir -p $W
B=ablation_rerun/occSbin; mkdir -p $B
CI=2; NC=8; RPT=2; AE=512; RU=4; TO=2400
for tile in 0 96; do
  for arm in global dsm resc; do
    case $arm in
      global) be=0; extra="";;
      dsm)    be=1; extra="";;
      resc)   be=2; extra="-DREAD_MODE=1 -DWRITE_MODE=1";;
    esac
    nvcc -arch=sm_90 resc_e2e.cu -o $B/${arm}_t${tile} -DBACKEND=$be -DRESERVE_ELEMS=8192 \
         -DSTRIDE_ELEMS=16384 -DTILE_KB=$tile -DCOMPUTE_ITERS=$CI $extra -lcudart 2>/dev/null
    patchelf --remove-rpath $B/${arm}_t${tile} 2>/dev/null
  done
done
echo SMALL_BUILDS_DONE
for spec in OFF60:cfg_h20_l2_60mb.config:0 ON60:cfg_h20_l2_60mb.config:1; do
  cfgname=${spec%%:*}; rest=${spec#*:}; cfgfile=${rest%%:*}; inv=${rest##*:}
  for tile in 0 96; do
    for arm in global dsm arm2 arm3; do
      case $arm in
        global) bin=global_t${tile}; G=1; MW=0;;
        dsm)    bin=dsm_t${tile};    G=1; MW=0;;
        arm2)   bin=resc_t${tile};   G=2; MW=0;;
        arm3)   bin=resc_t${tile};   G=3; MW=1;;
      esac
      d=$W/${cfgname}_t${tile}_${arm}; mkdir -p $d
      cp $cfgfile $d/gpgpusim.config
      printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
      ( cd $d && timeout $TO env RESC_GEN_RING=$G RESC_MW=$MW ../../../$B/$bin \
          cluster_size=8 repeat=$RPT num_clusters=$NC avg_elems=$AE burst_period=16 reuse_passes=$RU \
          > run.log 2>&1; echo $? > exit.code ) &
    done
  done
  wait
done
echo OCCS_ALL_DONE
