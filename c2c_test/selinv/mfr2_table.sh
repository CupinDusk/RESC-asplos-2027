#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Final reuse-axis table: advantage AND lead per R, both comparisons.
# Lead is reported alongside every ratio (addendum-02 S4): a rising ratio with a
# flat lead is arithmetic; a rising LEAD is mechanism.
P=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/mfr2
cyc(){ grep -a '^gpu_tot_sim_cycle' $1/run.log 2>/dev/null | tail -1 | awk '{print $NF}'; }
printf "%3s %10s %10s %10s %8s %10s %10s\n" R arm0@0 arm1@0 arm3@3 LEAD adv_vs_ca adv_vs_cg
for R in 1 2 4 8 16; do
  a0=$(cyc $P/k0_R${R}_arm0); a1=$(cyc $P/k0_R${R}_arm1); a3=$(cyc $P/k3_R${R}_arm3)
  if [ -z "$a0" ] || [ -z "$a1" ] || [ -z "$a3" ]; then
    printf "%3s %10s %10s %10s %8s %10s %10s\n" "$R" "${a0:--}" "${a1:--}" "${a3:--}" "-" "-" "-"
    continue
  fi
  awk -v R=$R -v a0=$a0 -v a1=$a1 -v a3=$a3 'BEGIN{
    base = (a1<a0)?a1:a0
    printf "%3s %10d %10d %10d %8d %10.4f %10.4f\n", R, a0, a1, a3, a1-a3, base/a3, a0/a3
  }'
done
