#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# MARKED-FRACTION SWEEP -- DIAGNOSTIC COLUMNS arm2@0 and arm2@1.
# Pre-registration addendum 02: markfrac-prereg-addendum-02.md, SHA-256 7a2d243d...,
# frozen 2026-09-03T13:51:06Z, while NO arm2 knob-0/knob-1 result existed.
#
# These are the KILL-W2 / W3 CONTROL COLUMNS: the same marked-registering arm paying a
# WHOLE-L1 acquire (knob 1) and no acquire (knob 0).  They are DIAGNOSTIC ONLY and are
# NOT admissible as a RESC result or as a baseline -- KILL-W1 is unchanged and is still
# computed as baselines=min(arm0@{0,1},arm1@{0,1}) vs RESC=min(arm2@{2,3},arm3@{2,3}).
#
# Waits for the main sweep so total concurrency never exceeds 3.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/selinv/mf/sweep
BIN=$T/selinv/mf/bin
NCL=2; NS=32; LAG=2; CL=4; TO=9000; MAXJOBS=3

while ! grep -q MF_SWEEP_ALL_DONE $T/selinv/mf_master2.log 2>/dev/null; do sleep 60; done
echo "main sweep done, starting arm2 diagnostic columns ($(date +%H:%M:%S))"

running_sims(){ ps -eo comm --no-headers | grep -c "^mf_arm"; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 10; done; }

for pv in 32 64 16 128 4 0 256 512 1024; do
  for inv in 0 1; do
    d=$W/k${inv}_v${pv}_arm2
    if [ -f $d/exit.code ] && [ "$(cat $d/exit.code)" = "0" ]; then echo "SKIP (done) $d"; continue; fi
    mkdir -p $d
    cp $T/cfg_h20_l2_60mb.config $d/gpgpusim.config
    printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" $inv >> $d/gpgpusim.config
    throttle
    echo "LAUNCH $d ($(date +%H:%M:%S))"
    ( cd $d && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=2 RESC_MW=0 \
        $BIN/mf_arm2 $NCL $NS $LAG $CL $pv > run.log 2>&1; echo $? > exit.code ) &
  done
done
wait
while [ "$(running_sims)" -gt 0 ]; do sleep 20; done
echo "MF_DIAG_DONE"
