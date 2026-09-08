#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# THE MAX-UNION ARENA TAX -- measurement runner.
#
# Pre-registration: resc-maxunion-prereg.md, SHA-256
#   878e8f28dd0f3d5d5b0bdee82941bf21f28342bb57f3e5230e4151462d39a902
#   frozen 2026-09-04T06:52:40Z, BEFORE any arm was executed under the simulator.
# Benchmark bound by hash: maxunion.cu, SHA-256
#   7b75a75ea96d853bffad00920996645803faef62cbd794ede39a4e218ba4c00c
#
# Arms (prereg S5).  ARM2 and ARM3 are THE SAME BINARY (mu_arm2); retention is
# chosen only by the environment, exactly as markfrac/pipe_chain do it.
#   A0 WORST_CASE_ARENA : mu_arm0, 13-page arena, SMEM staging (DSM incumbent)
#   A1 RECYCLING        : mu_arm1, 13-page arena, rotating slot (authors' escape)
#   A2 RESC             : mu_arm2 + RESC_RETAIN=1        (retention ON)
#   A3 RESC_NORETAIN    : mu_arm2, RESC_RETAIN unset     (retention OFF)
#   A4 BUCKETED_IDEAL   : mu_arm4, 2-page arena -- ILLEGAL when nbig>0, and that
#                         illegality is MEASURED via BITIDENTICAL=0, not asserted.
#
# CONFIG: cfg_h20_smem228.config -- cfg_h20.config with ONE line changed,
#   -gpgpu_shmem_per_block 49152  ->  233472
# per ADDENDUM 01 (resc-maxunion-addendum-01.md, SHA-256
# 66001f6f9c84c993d7c326324dc010cba1b6ec82756fde03862d3deff933bc0d, frozen
# 2026-09-04T07:00:49Z, written BEFORE any arm ran).  Reason: Stanford's verbatim
# arena is 13 pages x 16384 B = 212992 B/CTA, which EXCEEDS cfg_h20.config:115's
# per-block cap of 49152 B, so ARM0/ARM1/ARM4 could not launch as frozen.  48 KB
# is the STATIC limit; CC 9.0 allows 227 KB per block via DYNAMIC shared memory
# plus cudaFuncSetAttribute, which maxunion.cu already does.  cfg_h20.config:113
# already sets the per-SM carveout to 233472, so only the per-BLOCK line
# disagreed with it.  Nothing else in the config changes.
#
# ADDENDUM 01 A3 pre-registers the L1 split each arm must show (G4/KILL-M9),
# computed from shader.cc:3606-3616 with max_assoc=512, nset=4, line=128B:
#   ARM0/ARM1 (13pg, 212992 B) -> snaps to 228KB -> assoc  56 -> L1  28 KB
#   ARM4      ( 2pg,  32768 B) -> snaps to  32KB -> assoc 448 -> L1 224 KB
#   ARM2/ARM3 (RESC,      0 B) -> snaps to   0KB -> assoc 512 -> L1 256 KB
# If a run's assoc differs from this, KILL-M9 fires and NO cycle gap may be
# attributed to L1 capacity.
#
# cfg_h20_tinyL1.config is NOT used anywhere: it is defective (4/32 = 0 by
# integer division in get_max_cache_multiplier, gpu-cache.h:971-974; the assert
# is disabled by -DNDEBUG) and DEADLOCKS with retention OFF as well as ON, i.e.
# pre-existing.  See the retention design S8.2.
#
# CONCURRENCY CAP 2 -- the box runs at load ~70 with 25 users.
# NEVER broad-pkill: other users have jobs; exact PIDs only.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH

W=$T/maxunion/sweep
BIN=$T/maxunion/bin
mkdir -p $W $BIN

CFG=${CFG:-cfg_h20_smem228.config}
# Create it if absent: exactly one line differs from cfg_h20.config.
if [ ! -f "$T/$CFG" ]; then
  sed 's/^-gpgpu_shmem_per_block 49152/-gpgpu_shmem_per_block 233472/' \
      $T/cfg_h20.config > $T/$CFG
  # fail loudly if the substitution did not take
  grep -q "^-gpgpu_shmem_per_block 233472" $T/$CFG || {
    echo "FATAL: shmem_per_block substitution failed in $CFG"; exit 1; }
  echo "### created $CFG (diff vs cfg_h20.config):"
  diff $T/cfg_h20.config $T/$CFG
fi
# Fixed (prereg S7 as amended by ADDENDUM 02): cluster 8, 2 clusters = 16
# blocks, 64 thr, ntasks 32, and comm_lines = 64.
# comm_lines=64 is REQUIRED, not cosmetic: payload sizes are proportional to the
# pages a type needs (small = comm_ints, big = comm_ints*13/2), so
#   CL=64 -> small payload  8192 B, big payload 53248 B
#            ARM4 arena 32768 B  -> big payload OVERFLOWS -> bucketing ILLEGAL
#            ARM0 arena 212992 B -> big payload fits       -> incumbent LEGAL
# At CL<=32 the big payload still fits ARM4's 2-page arena, so ARM4 would report
# BITIDENTICAL=1 and KILL-M2 would fire on the BENCHMARK's construction rather
# than on the mechanism.  See ADDENDUM 02.
NCL=2; NT=32; CL=${CL:-64}
TO=${TO:-14400}
MAXJOBS=${MAXJOBS:-2}
BELIST="${BELIST:-4 2 8 0}"        # heterogeneity; 0 = degenerate homogeneous
PVLIST="${PVLIST:-256 0}"          # co-resident competitor present / absent
ARMS="${ARMS:-A0 A1 A2 A3 A4}"

running_sims(){ ps -eo comm --no-headers | grep -cE '^(mu_arm|rt_arm|mf_arm|m_R)'; }
throttle(){ while [ "$(running_sims)" -ge $MAXJOBS ]; do sleep 20; done; }

launch(){  # $1=arm $2=big_every $3=priv_lines
  local arm=$1 BE=$2 PV=$3 d bin RT
  RT=
  case $arm in
    A0) bin=mu_arm0;;
    A1) bin=mu_arm1;;
    A2) bin=mu_arm2; RT=1;;
    A3) bin=mu_arm2;;
    A4) bin=mu_arm4;;
    *)  echo "UNKNOWN ARM $arm"; return;;
  esac
  d=$W/${arm}_be${BE}_pv${PV}
  [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ] && return
  [ -f "$d/run.log" ] && [ ! -f "$d/exit.code" ] && { echo "IN-FLIGHT $d"; return; }
  mkdir -p "$d"
  cp $T/$CFG "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 3\n" >> "$d/gpgpusim.config"
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH $d ($(date +%H:%M:%S))"
  ( cd "$d" && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
      ${RT:+RESC_RETAIN=1 RESC_RETAIN_Q=100} \
      $BIN/$bin $NCL $NT $BE $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}

echo "### MAXUNION SWEEP cfg=$CFG arms=$ARMS BE=$BELIST PV=$PVLIST ($(date))"
echo "### cfg md5: $(md5sum $T/$CFG | awk '{print $1}')"
for b in 0 1 2 4; do
  [ -f "$BIN/mu_arm$b" ] && echo "### mu_arm$b md5: $(md5sum $BIN/mu_arm$b | awk '{print $1}')"
done
pass=0
while :; do
  pass=$((pass+1))
  echo "### pass $pass ($(date +%H:%M:%S))"
  for PV in $PVLIST; do
    for BE in $BELIST; do
      for arm in $ARMS; do launch $arm $BE $PV; done
    done
  done
  wait
  todo=0
  for PV in $PVLIST; do for BE in $BELIST; do for arm in $ARMS; do
    d=$W/${arm}_be${BE}_pv${PV}
    if [ ! -f "$d/exit.code" ] || [ "$(cat "$d/exit.code")" != "0" ]; then
      todo=$((todo+1)); fi
  done; done; done
  echo "### after pass $pass: $todo cells outstanding ($(date +%H:%M:%S))"
  [ "$todo" = "0" ] && break
  [ $pass -ge 12 ] && { echo "### giving up after $pass passes, $todo outstanding"; break; }
  sleep 30
done
echo "MAXUNION_SWEEP_DONE cfg=$CFG outstanding=$todo"
