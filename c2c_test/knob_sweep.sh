#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
echo "compscale,payload,cluster,arm,cycle"
# representative cells: (ae=8,cs=2)=32B/N1 loss; (ae=32,cs=4)=128B/N3 win; (ae=2048,cs=8)=8KB/N7 win
for tag in 1x 2x 3x; do
  cfg=$([ $tag = 1x ] && echo cfg_h20.config || echo cfg_h20_comp$tag.config)
  cp $cfg gpgpusim.config
  for spec in "8 2" "32 4" "2048 8"; do
    set -- $spec; ae=$1; cs=$2
    for arm in global dsm resc; do
      c=$(timeout 300 ./${arm}10240 cluster_size=$cs repeat=1 num_clusters=100 avg_elems=$ae burst_period=16 reuse_passes=4 2>/dev/null | grep gpu_tot_sim_cycle | tail -1 | grep -oE "[0-9]+")
      echo "$tag,$ae,$cs,$arm,$c"
    done
  done
done
echo "KNOB_SWEEP_DONE"
