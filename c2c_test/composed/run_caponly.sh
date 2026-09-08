#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Capacity-only baseline: the arm 4/5 r21 reviewers named as the deciding missing experiment.
# Prereg e8950bc1ac2650bc777ba9cba31350684b77e59243c380ea0d42af588018734d
# Source basis: gpu-cache.cc:1267-1308 -- selective_invalidate preserves a line on ONE predicate
# (selinv_addr_is_marked); no coherence state is consulted. gpu-cache.cc:460-472 -- the retention
# quota "can only reorder which of several evictable lines is evicted".
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
W=$T/c2c_test/composed/sweep_caponly
BINDIR=$T/c2c_test/composed/bin/v124
SRC=$T/c2c_test/composed/sweep_l2corrected
ARGS="2 32 4 64 256"
MAXJOBS=6
mkdir -p "$W"

running_sims() { pgrep -u hyren -f "cx_arm" | wc -l; }
throttle() { while [ "$(running_sims)" -ge "$MAXJOBS" ]; do sleep 30; done; }

# $1=cell $2=binary $3=source-config-dir $4=extra env assignments $5=knob override ("" = keep)
launch() {
  local nm="$1" arm="$2" srcdir="$3" envs="$4" knob="${5:-}"
  local d="$W/$nm"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm"; return; fi
  for p in $(pgrep -u hyren -f "cx_arm"); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && { echo "IN-FLIGHT $nm pid=$p"; return; }
  done
  throttle
  mkdir -p "$d"
  if [ -n "$knob" ]; then
    awk -v k="$knob" '/^-gpgpu_cluster_barrier_invalidate_l1 / { print "-gpgpu_cluster_barrier_invalidate_l1 " k; next } { print }' \
        "$srcdir/gpgpusim.config" > "$d/gpgpusim.config"
    grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL knob swap failed $nm"; return; }
  else
    cp "$srcdir/gpgpusim.config" "$d/gpgpusim.config"
  fi
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 $nm"; return; }
  rm -f "$d/exit.code"
  (
    cd "$d" || exit 1
    env RESC_QUIET=1 RESC_MW=1 $envs \
      timeout 21600 nice -n 10 "$BINDIR/$arm" $ARGS > run.log 2>&1
    echo $? > exit.code
  ) &
  echo "LAUNCHED $nm arm=$arm env='$envs'"
}

# C1 controls first: must reproduce the published cells instruction-identically.
launch CAP_A6    cx_arm6 "$SRC/C60_L1P_A6" "RESC_GEN_RING=3"
launch CAP_A2    cx_arm2 "$SRC/C60_L1P_A2" "RESC_GEN_RING=8"
# The reviewers' arm: knob 2 via mode override + forwarding off + NO retention.
launch CAP_A9    cx_arm2 "$SRC/C60_L1P_A2" "RESC_GEN_RING=8 RESC_NO_FWD=1" 2
# The sharper form: whole-L1 invalidate STILL charged, plus a pure replacement quota.
launch CAP_QONLY cx_arm6 "$SRC/C60_L1P_A6" "RESC_GEN_RING=3 RESC_RETAIN=1 RESC_RETAIN_Q=75"

wait
echo "=== DONE ==="
for nm in CAP_A6 CAP_A2 CAP_A9 CAP_QONLY; do
  d=$W/$nm
  ec=$( [ -f "$d/exit.code" ] && cat "$d/exit.code" || echo NONE )
  cyc=$(grep -m1 "gpu_sim_cycle =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  insn=$(grep -m1 "gpu_tot_sim_insn =" "$d/run.log" 2>/dev/null | awk '{print $3}')
  bit=$(grep -o "BITIDENTICAL=[01]" "$d/run.log" 2>/dev/null | head -1)
  knob=$(grep -o "parsed_knob=[0-9]*" "$d/run.log" 2>/dev/null | head -1)
  own=$(grep -o "owner_retained=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  wi=$(grep -o "whole_invalidates=[0-9]*" "$d/run.log" 2>/dev/null | tail -1)
  ret=$(grep -o "\[RESC_RETAIN\] enabled=[01] quota_pct=[0-9]*" "$d/run.log" 2>/dev/null | head -1)
  echo "CAPONLY $nm exit=$ec cyc=$cyc insn=$insn $bit $knob $own $wi | $ret"
done
