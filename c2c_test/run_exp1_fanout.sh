#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
W=exp1_fwd
echo "# Exp1 RESC fan-out sweep (special path = single-forwarder). small payload avg_elems=16 reuse=8"
echo "# N cluster cyc readshare_special readshare_normal cta checksum"
for cs in 2 4 8; do
  N=$((cs-1))
  d=$W/fo_${cs}; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
  ( cd $d
    ../resc_tf cluster_size=$cs repeat=4 num_clusters=40 avg_elems=16 burst_period=16 reuse_passes=8 > run.log 2>&1 )
  cyc=$(grep gpu_tot_sim_cycle $d/run.log | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  sp=$(grep -c "SUCCESS (special)" $d/run.log)
  no=$(grep -c "SUCCESS (normal)" $d/run.log)
  cta=$(grep -oE "CTA/core = [0-9]+" $d/run.log | head -1 | grep -oE "[0-9]+$")
  cksum=$(grep OUT_CHECKSUM $d/run.log | grep -oE "sum=[-0-9.]+" | head -1)
  echo "$N $cs ${cyc:-NA} $sp $no ${cta:-NA} ${cksum:-NA}"
done
echo EXP1_DONE
