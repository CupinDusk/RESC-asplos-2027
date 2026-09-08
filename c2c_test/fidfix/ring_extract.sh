#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/ring
printf "%-16s %10s %10s %12s %12s %14s %s\n" DIR CYCLES L2_RD_ACC DRAM_READS DRAM_WRITES LAGGED_SERVED CHECKSUM_OK
for d in l2_4mb_ring2 l2_4mb_ring3 l2_60mb_ring2 l2_60mb_ring3; do
  L=$d/run.log
  if [ ! -f "$L" ]; then printf "%-16s %10s\n" "$d" "PENDING"; continue; fi
  C=$(grep -E "^gpu_tot_sim_cycle = " $L | tail -1 | sed 's/.*= //')
  A=$(grep -E "^gpgpu_n_mem_read_global = " $L | tail -1 | sed 's/.*= //')
  R=$(grep -E "^total dram reads = " $L | tail -1 | sed 's/.*= //')
  W=$(grep -E "^total dram writes = " $L | tail -1 | sed 's/.*= //')
  G=$(grep -oE "lagged_reads_served=[0-9]+" $L | tail -1 | sed 's/.*=//')
  K=$(grep -E "PIPE_CHAIN" $L | tail -1 | grep -oE "BITIDENTICAL=[01]")
  printf "%-16s %10s %10s %12s %12s %14s %s\n" "$d" "${C:-NA}" "${A:-NA}" "${R:-NA}" "${W:-NA}" "${G:-NA}" "${K:-NA}"
done
