#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cp cfg_h20.config gpgpusim.config
for wl in hist mm; do
  for cs in 2 4 8; do
    ./valsweep/${wl}sim_c$cs 100000 512 > valsweep/${wl}_sim_c${cs}.log 2>&1
    cyc=$(grep "gpu_tot_sim_cycle" valsweep/${wl}_sim_c${cs}.log | tail -1 | sed "s/.*= *//")
    echo "${wl} c${cs} sim_cyc=${cyc}"
  done
done
echo "VALSWEEP_DONE"
