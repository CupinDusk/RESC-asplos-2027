#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
NC=${NC:-4}; NS=${NS:-32}; LAG=${LAG:-2}; TL=${TL:-4}; TO=${TO:-600}
echo "### pipe_overlap NC=$NC NS=$NS LAG=$LAG TL=$TL"
for G in "$@"; do
  out=$(timeout $TO env RESC_GEN_RING=$G ./pipe_overlap $NC $NS $LAG $TL 2>/dev/null)
  cyc=$(echo "$out" | grep "gpu_tot_sim_cycle =" | tail -1 | grep -oE "[0-9]+")
  gs=$(echo "$out" | grep RESC_GEN_STATS)
  pr=$(echo "$out" | grep PIPE_OVERLAP)
  bit=$(echo "$pr" | grep -oE "BITIDENTICAL=[0-9]+")
  served=$(echo "$gs" | grep -oE "readshare_served=[0-9]+")
  expired=$(echo "$gs" | grep -oE "readshare_expired=[0-9]+")
  ticks=$(echo "$gs" | grep -oE "epoch_ticks=[0-9]+")
  cs=$(echo "$pr" | grep -oE "OUT_CHECKSUM=[0-9]+")
  echo "G=$G cycles=${cyc:-TIMEOUT/NA} $ticks $served $expired $bit $cs"
done
