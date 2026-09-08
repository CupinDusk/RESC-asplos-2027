#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# RESC EPOCH-SCOPED RETENTION BIAS -- measurement runner.
# Pre-registration: resc-retention-prereg.md, SHA-256
#   99e938a03bbac246ac4238b8510d3b4190089b7a0ea52c1df0603cc65182ec1b
#   frozen 2026-09-04T05:11:22Z, BEFORE any retention binary was built or run.
#
# Arms (prereg S4).  Same binary throughout; discipline chosen only by env.
#   R0 reference : RESC_GEN_RING=3 RESC_MW=1, retention OFF
#   R1           : + RESC_RETAIN=1 RESC_RETAIN_Q=75
#   R2           : + RESC_RETAIN=1 RESC_RETAIN_Q=100   (strongest legal bias)
#   R3           : + RESC_RETAIN=1 RESC_RETAIN_Q=100 RESC_RETAIN_W=1 (release probe)
# Invalidate knob 3 (selective + owner-retaining) in every arm.
#
# Axis A (marked footprint / L1 capacity) is swept by comm_lines CL.
# Axis B (unmarked competition)          is swept by priv_lines PV in {0,256}.
# L1 capacity is set by CFG; the RATIO is computed from each run's own log by
# extract_retain.sh, never asserted here.
#
# CONCURRENCY CAP 3, counted across every retention/markfrac sim on the shared
# box.  NEVER broad-pkill: other users have jobs.
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

# Fixed geometry (prereg S3): cluster 8, 2 clusters = 16 blocks, 64 thr, lag 2.
NCL=2; NS=32; LAG=2
# Small-L1 config so the marked footprint can exceed L1 capacity cheaply
# (prereg S5 feasibility rule -- this IS a config change and is reported).
CFG=${CFG:-cfg_h20_tinyL1.config}
TO=${TO:-14400}
MAXJOBS=3
CLLIST="${CLLIST:-1 2 4 8 16 32}"
PVLIST="${PVLIST:-0 256}"
ARMS="${ARMS:-R0 R1 R2 R3}"

running_sims(){ ps -eo comm --no-headers | grep -cE '^(mf_arm|mfr_arm|m_R|rt_arm)'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 15; done; }

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
for PV in $PVLIST; do
  for CL in $CLLIST; do
    for arm in $ARMS; do launch $arm $CL $PV; done
  done
done
wait
echo "RETAIN_SWEEP_DONE cfg=$CFG"
