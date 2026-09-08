#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Boundary probe: where does arm3 (RESC_MW=1) survive vs deadlock as contention rises?
# All nclusters=1 to be fast under host load. 2>&1 so we catch the deadlock banner.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
TO=150
probe(){ # K skew UPD block label
  local K=$1 SK=$2 UPD=$3 BLK=$4 L=$5
  out=$(RESC_MW=1 timeout $TO ./hist_mw global $K $SK $UPD 1 $BLK 2>&1)
  ec=$?
  cyc=$(echo "$out"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+")
  bit=$(echo "$out"|grep MW_HIST|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2)
  dead=$(echo "$out"|grep -c "deadlock detected")
  tok=$(echo "$out"|grep RESC_MW_STATS|tail -1|grep -oE "atomic_upgradereq=[0-9]+"|cut -d= -f2)
  printf "%-28s exit=%-4s deadlock=%s cyc=%-8s bit=%-3s tokens=%s\n" "$L" "$ec" "$dead" "${cyc:-NA}" "${bit:-NA}" "${tok:-NA}"
}
echo "=== arm3 RESC_MW=1 deadlock boundary probe (nclusters=1) ==="
probe 256 0 1  64 "UPD1_K256_uniform_blk64"
probe 256 0 1   1 "UPD1_K256_uniform_blk1"
probe 256 0 4   1 "UPD4_K256_uniform_blk1"
probe 256 0 8   1 "UPD8_K256_uniform_blk1"
probe 64  0 8   1 "UPD8_K64_uniform_blk1"
probe 64  2 8   1 "UPD8_K64_zipf1.5_blk1"
probe 256 0 64  1 "UPD64_K256_uniform_blk1"
echo "PROBE_DONE"
