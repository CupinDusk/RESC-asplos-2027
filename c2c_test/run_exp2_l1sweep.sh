#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Exp2: marked-line L1-hit-cost sweep. Knob = -gpgpu_l1_latency (base 41 -> +4=45,+8=49).
# In the comm-only legality benchmark (TILE=0) nearly all L1 hits are reused
# marked-line hits; cg baseline uses ld.cg (bypasses L1) so it is ~insensitive.
# 8KB payload (ae=2048), reuse=8, equal occupancy (TILE=0), cluster 8 (N=7).
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
W=exp2_l1
echo "# Exp2 L1-hit-cost sweep  pb=8192 reuse=8 cluster=8 N=7 NC=40 repeat=2"
echo "# l1lat arm cyc cta checksum"
for cfg in cfg_h20.config:41 cfg_l1_45.config:45 cfg_l1_49.config:49; do
  conf=${cfg%%:*}; lat=${cfg##*:}
  for arm in cg resc; do
    d=$W/s_${lat}_${arm}; mkdir -p $d; cp $conf $d/gpgpusim.config
    ( cd $d
      ../${arm}_l cluster_size=8 repeat=2 num_clusters=40 avg_elems=2048 burst_period=16 reuse_passes=8 > run.log 2>&1 ) &
  done
done
wait
for cfg in cfg_h20.config:41 cfg_l1_45.config:45 cfg_l1_49.config:49; do
  lat=${cfg##*:}
  for arm in cg resc; do
    d=$W/s_${lat}_${arm}
    cyc=$(grep gpu_tot_sim_cycle $d/run.log | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
    cta=$(grep -oE "CTA/core = [0-9]+" $d/run.log | head -1 | grep -oE "[0-9]+$")
    ck=$(grep OUT_CHECKSUM $d/run.log | grep -oE "sum=[-0-9.]+" | head -1)
    echo "$lat $arm ${cyc:-NA} ${cta:-NA} ${ck:-NA}"
  done
done
echo EXP2_DONE
