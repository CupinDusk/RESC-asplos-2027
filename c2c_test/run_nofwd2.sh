#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# A8 = A2 + RESC_NO_FWD=1. Prereg c72a076e...
# ISOLATED WORKDIR: writes ONLY under composed/sweep_nofwd, never the shared composed/sweep.
# L1P_A8 is already running under the old shared path and is left alone; this driver
# handles OCC only, and will pick up L1P from the old location for reporting.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/composed/sweep_nofwd; BIN=$T/composed/bin/v124
CFG=cfg_h20_smem228.config
TO=${TO:-21600}
MAXJOBS=${MAXJOBS:-3}
running_sims(){ ps -u hyren -o comm --no-headers | grep -c "^cx_arm" ; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 20; done; }
launch(){
  local pn=$1; local args="$2"; local d="$W/${pn}_A8"
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && { echo "DONE $d"; return; }
  if [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ]; then
    local alive=0
    for p in $(pgrep -u hyren -f cx_arm2); do
      [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && alive=1
    done
    [ $alive -eq 1 ] && { echo "IN-FLIGHT(verified live) $d"; return; }
    echo "STALE $d -- run.log with no live process; re-running"
  fi
  throttle
  mkdir -p "$d"; cp $T/$CFG "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %d\n" 3 >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  echo "LAUNCH $d NO_FWD=1 knob=3 args=\"$args\" ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
      RESC_RETAIN=1 RESC_RETAIN_Q=100 RESC_NO_FWD=1 \
      $BIN/cx_arm2 $args > run.log 2>&1; echo $? > exit.code ) &
}
echo "### A8 ISOLATED  workdir=$W  ($(date))"
pass=0
while :; do
  pass=$((pass+1)); echo "### pass $pass ($(date +%H:%M:%S))"
  launch OCC "2 32 4 64 1024"
  sleep 90
  [ -f "$W/OCC_A8/exit.code" ] && [ "$(cat $W/OCC_A8/exit.code)" = "0" ] && { echo "### OCC DONE"; break; }
  [ $pass -ge 200 ] && { echo "### GIVING UP"; break; }
done
