#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# THE DISCRIMINATOR BASELINE: does the strongest legal SMEM-staging idiom beat RESC?
# Prereg 8a558add82e0c38f1f93690a3979cd80d9f08cfb773cd814bd6455c87d2eb3d2
#
# r27 sim-method W3: "necessity is CONSTRUCTED rather than established: on real silicon the
#   sync_restrict fence is a single NOP, so an SMEM-staging idiom is immune to the tax RESC removes."
# ARM=4 (BUCKETED_IDEAL) IS that idiom and maxunion.cu built it for exactly this separation.
# big_every=0 => homogeneous => ARM4's small arena is LEGAL (no big payload to overflow it).
# NOTE: reuses the 2026-09-04 binaries; NO rebuild, so the program cannot drift.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
T=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
cd $T || exit 1
export PATH=$T/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=$T/maxunion/sweep
BIN=$T/maxunion/bin
CFG=cfg_h20_smem228.config
[ -f "$T/$CFG" ] || { echo "FATAL missing $CFG"; exit 1; }
DL2_960='-gpgpu_cache:dl2 S:960:128:16,L:B:m:L:L,A:192:4,32:0,32'
NCL=2; NT=32; BE=0; CL=64; PV=256
TO=14400; MAXJOBS=2

for b in mu_arm0 mu_arm2 mu_arm4; do
  [ -x "$BIN/$b" ] || { echo "FATAL missing binary $BIN/$b"; exit 1; }
  echo "REUSE $b md5=$(md5sum $BIN/$b | awk '{print $1}')"
done

running(){ ps -eo comm --no-headers | grep -cE '^mu_arm'; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }

launch(){
  local arm=$1
  local bin=$2
  local knob=$3
  local RT=$4
  local d=$W/D_${arm}_be0_pv${PV}
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $arm"; return; fi
  mkdir -p "$d"
  awk -v l2="$DL2_960" '/^-gpgpu_cache:dl2 / { print l2; next } { print }' "$T/$CFG" > "$d/gpgpusim.config"
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 %s\n" "$knob" >> "$d/gpgpusim.config"
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 $arm"; return; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL knob $arm"; return; }
  grep -q "^-gpgpu_shmem_per_block 233472" "$d/gpgpusim.config" || { echo "FATAL smem cap $arm"; return; }
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH D_$arm knob=$knob RT=${RT:-off}"
  ( cd "$d" && timeout $TO env RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
      ${RT:+RESC_RETAIN=1 RESC_RETAIN_Q=100} \
      $BIN/$bin $NCL $NT $BE $CL $PV > run.log 2>&1; echo $? > exit.code ) &
}

launch ARM4 mu_arm4 1 ""      # the discriminator: strongest legal SMEM staging
launch ARM2 mu_arm2 3 "1"     # RESC (control)
launch ARM0 mu_arm0 1 ""      # DSM worst-case arena, for scale
wait

echo "=== DONE ==="
for arm in ARM4 ARM2 ARM0; do
  d=$W/D_${arm}_be0_pv${PV}; L=$d/run.log
  printf "DISCRIM %-5s exit=%-4s cyc=%-9s insn=%-11s %s %s %s knob=%s %s\n" "$arm" \
    "$([ -f $d/exit.code ] && cat $d/exit.code || echo NONE)" \
    "$(grep -m1 'gpu_sim_cycle =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -m1 'gpu_tot_sim_insn =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -oE 'BITIDENTICAL=[01]' $L 2>/dev/null | tail -1)" \
    "$(grep -oE 'SMEM_PER_CTA_B=[0-9]+' $L 2>/dev/null | tail -1)" \
    "$(grep -oE '(whole|selective)_invalidates=[0-9]+' $L 2>/dev/null | tail -1)" \
    "$(grep -oE '^-gpgpu_cluster_barrier_invalidate_l1 [0-9]+' $d/gpgpusim.config 2>/dev/null | grep -oE '[0-9]+$')" \
    "$(grep -m1 -oE 'Reconfigure L1 cache to [0-9]+KB' $L 2>/dev/null)"
done
