#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# R=1 PER-ARM GATE CHECK.
# Pre-registration: markfrac-reuse-prereg-addendum-01.md, SHA-256 73b889e6b39d69ca...,
# frozen 2026-09-03T18:06:17Z.  Reference values companion: ...-amended.md (32d55620...).
#
# THE GATE (addendum S3): at R=1, for EVERY ONE of the eight arm/knob cells, the
# compile-time-R build must match markfrac.cu on BOTH gpu_tot_sim_insn AND
# gpu_tot_sim_cycle -- EXACTLY.  No tolerance.  Any mismatch in any arm => family VOID.
#
# References are read live from the completed markfrac sweep (selinv/mf/sweep/k*_v32_*),
# not hardcoded, so the comparison cannot drift from the actual published cells.
P=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv
pass=0; fail=0; pend=0
printf "%-9s %-22s %-22s %s\n" "cell" "reference (cyc/insn)" "R=1 build (cyc/insn)" "verdict"
for spec in arm0:0 arm0:1 arm1:0 arm1:1 arm2:2 arm2:3 arm3:2 arm3:3; do
  a=${spec%:*}; k=${spec#*:}
  ref=$P/mf/sweep/k${k}_v32_${a}
  new=$P/mfr2/k${k}_R1_${a}
  rc=$(grep -a '^gpu_tot_sim_cycle' $ref/run.log 2>/dev/null | tail -1 | awk '{print $NF}')
  ri=$(grep -a '^gpu_tot_sim_insn'  $ref/run.log 2>/dev/null | tail -1 | awk '{print $NF}')
  if [ ! -f $new/exit.code ]; then
    printf "%-9s %-22s %-22s %s\n" "${a}@${k}" "${rc}/${ri}" "-" "PENDING"
    pend=$((pend+1)); continue
  fi
  ec=$(cat $new/exit.code)
  nc=$(grep -a '^gpu_tot_sim_cycle' $new/run.log 2>/dev/null | tail -1 | awk '{print $NF}')
  ni=$(grep -a '^gpu_tot_sim_insn'  $new/run.log 2>/dev/null | tail -1 | awk '{print $NF}')
  if [ "$ec" != "0" ]; then
    v="FAIL(exit=$ec)"; fail=$((fail+1))
  elif [ "$nc" = "$rc" ] && [ "$ni" = "$ri" ]; then
    v="PASS"; pass=$((pass+1))
  else
    v="FAIL"; fail=$((fail+1))
    [ "$nc" != "$rc" ] && v="$v cyc:$((nc-rc))"
    [ "$ni" != "$ri" ] && v="$v insn:$((ni-ri))"
  fi
  printf "%-9s %-22s %-22s %s\n" "${a}@${k}" "${rc}/${ri}" "${nc}/${ni}" "$v"
done
echo
echo "pass=$pass fail=$fail pending=$pend"
if [ $pend -gt 0 ]; then echo "GATE INCOMPLETE"
elif [ $fail -gt 0 ]; then echo "GATE FAILED -- family VOID, do not run R>=2"
else echo "GATE PASSED -- all 8 arms exact on both metrics; R>=2 may proceed"; fi
exit 0
