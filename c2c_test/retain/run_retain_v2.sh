#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# RESC EPOCH-SCOPED RETENTION BIAS -- measurement runner (CORRECTED CONFIG).
#
# Pre-registration: resc-retention-prereg.md, SHA-256
#   99e938a03bbac246ac4238b8510d3b4190089b7a0ea52c1df0603cc65182ec1b (FROZEN).
# Arms R0-R3 exactly as prereg S4.  Guards computed by extract_retain.sh (S6).
#
# ============================================================================
# DEVIATION FROM THE PREREG, DECLARED: cfg_h20_tinyL1.config is NOT used.
# ============================================================================
# The prereg S5 names cfg_h20_tinyL1.config to reach a high marked-footprint
# ratio cheaply.  That config is DEFECTIVE and the defect is MEASURED, not
# assumed:
#   * tinyL1 sets -gpgpu_unified_l1d_size 4 against a base
#     -gpgpu_cache:dl1 S:4:128:64 whose own size is 4*64*128/1024 = 32 KB.
#   * get_max_cache_multiplier() (gpu-cache.h:971-974) computes 4/32 = 0 by
#     INTEGER division, giving max_assoc = 0, i.e. a degenerate L1.
#   * the assert() that would have caught this is disabled by -DNDEBUG in the
#     Release build (build/CMakeCache.txt:78).
#   * the simulator DEADLOCKS on it: exit 134,
#     "GPGPU-Sim uArch: ERROR ** deadlock detected".
#   * DECISIVE CONTROL: it deadlocks IDENTICALLY with retention OFF
#     (retain/diag_tinyL1_R0, RESC_RETAIN unset, exit 134).  The defect is
#     PRE-EXISTING and NOT caused by the retention patch.
# cfg_h20_smallL1.config (unified 16) has the same defect (16/32 = 0).
# The smallest LEGAL small-L1 config is unified = 32 KB (multiplier 1,
# assoc 64), created here as cfg_h20_L1_32KB.config.  Every arm at a point uses
# the identical config, and the config is reported per point.
#
# Axis A bracketing on L1 = 32 KB (prereg S5 wants >=2 points below 0.5x, one
# near 1x, >=2 above 2x).  Protected footprint per SM ~ W(=3) * 2 tiles *
# comm_lines * 128 B; the RATIO is recomputed by extract_retain.sh from each
# run's OWN reported geometry, never from this comment:
#   CL=1   ->   768 B  ratio 0.023
#   CL=4   ->  3072 B  ratio 0.094   } two points well below 0.5x
#   CL=16  -> 12288 B  ratio 0.375
#   CL=32  -> 24576 B  ratio 0.750   } bracketing 1x
#   CL=64  -> 49152 B  ratio 1.500
#   CL=128 -> 98304 B  ratio 3.000   } two points above 2x (with CL=64)
#
# CONCURRENCY CAP 2 (the box is at load ~70 with 25 users; the prereg allows 3,
# and 2 is narrower, which is permitted).  NEVER broad-pkill: exact PIDs only.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH

W=$T/retain/sweep
BIN=$T/retain/bin
mkdir -p $W $BIN

# Create the LEGAL small-L1 config if absent (unified 32 KB = multiplier 1).
CFG=${CFG:-cfg_h20_L1_32KB.config}
if [ ! -f "$T/$CFG" ]; then
  sed 's/-gpgpu_unified_l1d_size 256/-gpgpu_unified_l1d_size 32/' \
      $T/cfg_h20.config > $T/$CFG
fi

# Fixed geometry (prereg S3): cluster 8, 2 clusters = 16 blocks, 64 thr, lag 2.
NCL=2; NS=32; LAG=2
TO=${TO:-14400}
MAXJOBS=${MAXJOBS:-2}
CLLIST="${CLLIST:-1 4 16 32 64 128}"
# Axis B extended per RETENTION ADDENDUM 06 A1 (SHA-256
# c11c92cde6951a562575ee4d7162b1bb98d59246f6b22d09d9119ef0eb3a5c11), written
# BEFORE these runs.  REASON, measured not assumed: the frozen prereg sized
# priv_lines=256 as "32 KB against a 256 KB L1" (12.5%).  At the LEGAL
# replacement config the L1 is 32 KB, so priv_lines=256 is 100% of L1 -- the
# CL=64 control measured unmarked_hitrate = EXACTLY 0.000000, i.e. total thrash.
# Axis B as frozen therefore offered only "competitor absent" (0%) or
# "competitor saturating" (100%), with no point where the competitor is present
# but the L1 still functions -- the only regime where a retention BIAS (as
# opposed to a pin) could plausibly pay.  Both frozen points are RETAINED;
# three are added:  0 -> 0%   32 -> 12.5% (the proportion the prereg intended)
# 64 -> 25%   128 -> 50%   256 -> 100% (frozen, retained).
PVLIST="${PVLIST:-0 32 64 128 256}"
ARMS="${ARMS:-R0 R1 R2 R3}"

running_sims(){ ps -eo comm --no-headers | grep -cE '^(mf_arm|mfr_arm|m_R|rt_arm|mu_arm)'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 20; done; }

launch(){  # $1=arm $2=comm_lines $3=priv_lines
  local arm=$1 CL=$2 PV=$3 d G MW RT RQ RW
  G=3; MW=1; RT=; RQ=; RW=
  case $arm in
    R0) ;;                                  # retention OFF (knob unset)
    R1) RT=1; RQ=75;;
    R2) RT=1; RQ=100;;
    R3) RT=1; RQ=100; RW=1;;
    *)  echo "UNKNOWN ARM $arm"; return;;
  esac
  d=$W/${arm}_cl${CL}_pv${PV}
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && return
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p "$d"
  cp $T/$CFG "$d/gpgpusim.config"
  # invalidate knob 3 = marked-line selective + owner-retaining, every arm
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH $d ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=$G RESC_MW=$MW \
      ${RT:+RESC_RETAIN=$RT} ${RQ:+RESC_RETAIN_Q=$RQ} ${RW:+RESC_RETAIN_W=$RW} \
      $BIN/rt_arm2 $NCL $NS $LAG $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}

echo "### RETAIN SWEEP cfg=$CFG arms=$ARMS CL=$CLLIST PV=$PVLIST ($(date))"
echo "### cfg md5: $(md5sum $T/$CFG | awk '{print $1}')"
echo "### so md5: $(md5sum /legacy-research/owner/cudatest/RESC-c2c-wt/build/lib/libcudart.so.12 | awk '{print $1}')"
# ---------------------------------------------------------------------------
# PRIORITISED ORDER (prereg S5 feasibility rule: declared in advance so it
# cannot become a silent choice).  The full grid is 4 arms x 6 CL x 5 PV = 120
# cells; at ~40 min/cell measured (CL=64 took 42 min) and concurrency 2 that is
# ~40 h, which is not feasible.  Cells are therefore ordered so the
# DECISION-RELEVANT ones complete first, and the run is resumable (a completed
# cell is never redone), so an interruption costs only the tail:
#
#   TIER 1 -- the answer to Q2 (does retention buy anything?).  R0 vs R2 at the
#             points where the bias is known to engage, at the competitor level
#             the prereg actually intended (PV=32 = 12.5% of L1) plus the
#             saturating one it names (PV=256).  CL=128 is the over-capacity
#             control point (1.50x L1); CL=64 is the under-quota point already
#             measured for R2.
#   TIER 2 -- the cost CURVE along CL (Q1) at the intended competitor level.
#   TIER 3 -- R1 (Q=75) and R3 (W=1, the self-release probe, prereg P4).
#   TIER 4 -- the remaining PV levels, filling out Axis B.
#
# Every tier uses the identical binary and config; nothing about a cell changes
# with the tier it sits in.  Whatever is unfinished at report time is listed as
# unfinished, with its reason -- never silently dropped.
# ---------------------------------------------------------------------------
run_tier(){ # $1=label, then triples "arm:CL:PV"
  local label=$1; shift
  echo "### TIER $label ($(date +%H:%M:%S))"
  for spec in "$@"; do
    launch "${spec%%:*}" "$(echo $spec|cut -d: -f2)" "${spec##*:}"
  done
  wait
}

if [ "${PRIORITISED:-1}" = "1" ]; then
  # TIER 1: the R0-vs-R2 comparison at the engaging points
  run_tier 1 R0:128:32 R2:128:32 R0:64:32 R2:64:32 R0:128:256 R2:128:256
  # TIER 2: cost curve along CL at the intended competitor level
  run_tier 2 R0:1:32 R2:1:32 R0:4:32 R2:4:32 R0:16:32 R2:16:32 R0:32:32 R2:32:32
  # TIER 3: quota variant and the self-release probe where the bias engages
  run_tier 3 R1:128:32 R3:128:32 R1:64:32 R3:64:32
  # TIER 4: remaining Axis B levels for the two headline arms
  for PV in 0 64 128 256; do
    run_tier "4-pv$PV" R0:128:$PV R2:128:$PV R0:64:$PV R2:64:$PV
  done
fi

pass=0
while :; do
  pass=$((pass+1))
  echo "### pass $pass ($(date +%H:%M:%S))"
  for PV in $PVLIST; do
    for CL in $CLLIST; do
      for arm in $ARMS; do launch $arm $CL $PV; done
    done
  done
  wait
  todo=0
  for PV in $PVLIST; do for CL in $CLLIST; do for arm in $ARMS; do
    d=$W/${arm}_cl${CL}_pv${PV}
    if [ ! -f "$d/exit.code" ] || [ "$(cat "$d/exit.code")" != "0" ]; then
      todo=$((todo+1)); fi
  done; done; done
  echo "### after pass $pass: $todo cells outstanding ($(date +%H:%M:%S))"
  [ "$todo" = "0" ] && break
  [ $pass -ge 12 ] && { echo "### giving up after $pass passes, $todo outstanding"; break; }
  sleep 30
done
echo "RETAIN_SWEEP_DONE cfg=$CFG outstanding=$todo"
