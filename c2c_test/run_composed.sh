#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# COMPOSED-REGIME SWEEP (legality ^ capacity, one kernel, one build).
# Prereg: composed-regime-prereg.md SHA-256
#   729c2583948af7a3a8683a494ac785271034ab855d0ceb2744ee4a151b0253cb
# Benchmark: composed.cu SHA-256
#   7d808dc83912d5236545f7ad9089ced9d12cc5802e8f616c6aeb28e0990a5222
#
# ARM -> simulator invalidate knob (-gpgpu_cluster_barrier_invalidate_l1).
# The knob is HOW the epoch-boundary acquire is charged; see composed.cu header.
# Modes 2/3 model RESC HARDWARE and are barred from every software arm.
#   A0 worstcase_arena_DSM  knob 1  DSM + real whole-L1 CCTL.IVALL
#   A2 RESC_retain          knob 3  RESC selective + owner-retaining (RESC HW)
#   A5 cg_nostage           knob 0  .cg bypasses L1 -> needs NO acquire
#   A6 ca_acquire           knob 1  .ca caches -> needs whole-L1 invalidate
#   A7 hybrid_restrict      knob 0  SMEM-staged + sync_restrict escape
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH

W=$T/composed/sweep; BIN=$T/composed/bin/v124; mkdir -p $W
CFG=cfg_h20_smem228.config
TO=${TO:-21600}
MAXJOBS=${MAXJOBS:-2}
ARMS="${ARMS:-A0 A2 A5 A6 A7}"
# Pressure points: name:NCL:NT:BE:CL:PV
PTS="${PTS:-LOW:2:32:0:64:0 L1P:2:32:4:64:256 OCC:2:32:4:64:1024}"

# Count ONLY my own composed binaries. Never broad-pkill: shared box, 24 users.
running_sims(){ ps -u hyren -o comm --no-headers | grep -c '^cx_arm' ; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 20; done; }

launch(){ # $1=arm $2=pointname $3=NCL $4=NT $5=BE $6=CL $7=PV
  local arm=$1 pn=$2 NCL=$3 NT=$4 BE=$5 CL=$6 PV=$7 d bin knob RT
  RT=
  case $arm in
    A0) bin=cx_arm0; knob=1;;
    A2) bin=cx_arm2; knob=3; RT=1;;
    A5) bin=cx_arm5; knob=0;;
    A6) bin=cx_arm6; knob=1;;
    A7) bin=cx_arm7; knob=0;;
    *) echo "UNKNOWN ARM $arm"; return;;
  esac
  d=$W/${pn}_${arm}
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && return
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p "$d"
  cp $T/$CFG "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %d\n" $knob >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH $d knob=$knob args=\"$NCL $NT $BE $CL $PV\" ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
      ${RT:+RESC_RETAIN=1 RESC_RETAIN_Q=100} \
      $BIN/$bin $NCL $NT $BE $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}

echo "### COMPOSED SWEEP cfg=$CFG arms=$ARMS ($(date))"
echo "### cfg md5: $(md5sum $T/$CFG | awk '{print $1}')"
echo "### .so md5: $(md5sum /legacy-research/owner/cudatest/RESC-c2c-wt/build/lib/libcudart.so.12 | awk '{print $1}')"
for a in 0 2 5 6 7; do
  [ -f "$BIN/cx_arm$a" ] && echo "### cx_arm$a md5: $(md5sum $BIN/cx_arm$a | awk '{print $1}')"
done
pass=0
while :; do
  pass=$((pass+1)); echo "### pass $pass ($(date +%H:%M:%S))"
  for p in $PTS; do
    IFS=: read pn NCL NT BE CL PV <<< "$p"
    for arm in $ARMS; do launch $arm $pn $NCL $NT $BE $CL $PV; done
  done
  wait
  todo=0
  for p in $PTS; do
    IFS=: read pn NCL NT BE CL PV <<< "$p"
    for arm in $ARMS; do
      d=$W/${pn}_${arm}
      { [ ! -f "$d/exit.code" ] || [ "$(cat "$d/exit.code")" != "0" ]; } && todo=$((todo+1))
    done
  done
  echo "### after pass $pass: $todo outstanding ($(date +%H:%M:%S))"
  [ "$todo" = "0" ] && break
  [ $pass -ge 6 ] && { echo "### giving up after $pass passes, $todo outstanding"; break; }
  sleep 30
done
echo "COMPOSED_SWEEP_DONE outstanding=$todo"
