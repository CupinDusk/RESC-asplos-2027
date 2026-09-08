#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Ground truth for "is the marked-fraction sweep finished?"
#
# Does NOT rely on any runner printing a DONE marker.  The diag and repair passes each
# end with `while running_sims > 0; do sleep 20; done`, and running_sims counts EVERY
# mf_arm process on the box -- including other passes' jobs.  So MF_DIAG_DONE /
# MF_REPAIR_DONE can be starved indefinitely while work is still progressing normally.
# Completion is therefore defined by THE DATA: all 90 cells present with exit 0.
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/mf/sweep
ok=0; nmiss=0; bad=""; missing=""
for pv in 0 4 16 32 64 128 256 512 1024; do
  for spec in arm0:0 arm0:1 arm1:0 arm1:1 arm2:2 arm2:3 arm3:2 arm3:3 arm2:0 arm2:1; do
    a=${spec%:*}; k=${spec#*:}
    d=$W/k${k}_v${pv}_${a}
    if [ -f "$d/exit.code" ]; then
      c=$(cat "$d/exit.code")
      if [ "$c" = "0" ]; then ok=$((ok+1)); else bad="$bad k${k}_v${pv}_${a}=$c"; fi
    else
      nmiss=$((nmiss+1)); missing="$missing k${k}_v${pv}_${a}"
    fi
  done
done
run=$(ps -eo comm --no-headers | grep -c '^mf_arm')
anchor=0
for spec in arm0:0 arm0:1 arm1:0 arm1:1 arm2:2 arm2:3 arm3:2 arm3:3 arm2:0 arm2:1; do
  a=${spec%:*}; k=${spec#*:}
  d=$W/k${k}_v32_${a}
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && anchor=$((anchor+1))
done
echo "complete=$ok/90 anchor=$anchor/10 running=$run missing=$nmiss"
[ -n "$bad" ] && echo "FAILED:$bad"
[ "$ok" = "90" ] && echo "SWEEP_COMPLETE"
# stall detector: nothing running and cells still missing => every runner has exited
if [ "$run" = "0" ] && [ "$ok" != "90" ]; then
  echo "STALLED: no simulator running but $nmiss cells missing --$missing"
fi
exit 0
