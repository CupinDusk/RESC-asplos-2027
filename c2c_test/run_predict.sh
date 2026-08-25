#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_cal195.config gpgpusim.config   # FROZEN calibrated config (lat=195)
mkdir -p predict
echo "=== HELD-OUT PREDICT: c8, reuse_passes=1 (one-shot matched), frozen lat=195 START $(date) ==="
# span the silicon crossover ~384 floats: test 128..768
for nf in 128 192 256 384 512 768; do
  for arm in global dsm; do
    f=predict/${arm}_c8_nf${nf}.log
    ./${arm}10240 cluster_size=8 repeat=1 num_clusters=100 avg_elems=$nf burst_period=16 reuse_passes=1 > "$f" 2>&1
    echo "$arm nf=$nf cyc=$(grep gpu_tot_sim_cycle "$f" | tail -1 | grep -oE '[0-9]+')"
  done
done
echo "PREDICT_DONE"
