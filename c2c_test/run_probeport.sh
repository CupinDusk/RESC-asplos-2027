#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# SUPPLIER TAG-ARRAY PROBE-PORT OCCUPANCY SWEEP -- the cost main.tex:1121 admits is not charged.
# Prereg be694aa96f9ccd40cc26fd3f88a64a20daeb087ce00249d2ae505a12d29ac84e (frozen before the patch).
#
# Control = C60_L1P_A2 (corrected S:960, knob 3, G=8) = 1,227,505 cyc / insn 80,818,560.
# Opponent = C60_L1P_A6 (S:960, knob 1) = 2,273,856 -- has NO probes, so it must not move.
# ⚠️ The paper's 1.7710x pair mixes L2 sizes (published L1P_A2 is S:64); this sweep uses the
#    MATCHED corrected pair (1,227,505 vs 2,273,856 = 1.8524x). One knob at a time.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
C=$T/c2c_test/composed
export PATH=$T/c2c_test/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
BIN=$C/bin
W=$C/sweep_probeport
mkdir -p "$W"
ARGS="2 32 4 64 256"     # L1P point: NCL NT BE CL PV  (from run_composed.sh PTS "L1P:2:32:4:64:256")
MAXJOBS=3

echo "### sim .so md5: $(md5sum $T/build/lib/libcudart.so.12 | awk '{print $1}')  (pre-patch was 0a27b8aac382e749c66569ff706014f0)"
for b in cx_arm2 cx_arm6; do
  [ -x "$BIN/$b" ] || { echo "FATAL missing $BIN/$b"; exit 1; }
  echo "### REUSE $b md5=$(md5sum $BIN/$b | awk '{print $1}')"
done

running(){ ps -eo comm --no-headers | grep -cE '^cx_arm'; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }

# $1=cell $2=binary $3=knob $4=probe_port_cycles
launch(){
  local nm="$1"
  local bin="$2"
  local knob="$3"
  local pp="$4"
  local d="$W/$nm"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm"; return; fi
  mkdir -p "$d"
  # copy the matching published cell's own config: knob 3 -> A2, knob 1 -> A6
  local srccell=A2
  [ "$knob" = "1" ] && srccell=A6
  cp "$C/sweep_l2corrected/C60_L1P_$srccell/gpgpusim.config" "$d/gpgpusim.config" || { echo "FATAL no src config $srccell"; return; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL src config $srccell knob mismatch"; return; }
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960 $nm"; return; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL knob $nm"; return; }
  rm -f "$d/exit.code"
  throttle
  echo "LAUNCH $nm knob=$knob RESC_PROBE_PORT=$pp"
  # A2 (knob 3) REQUIRES RESC_RETAIN=1 RESC_RETAIN_Q=100 -- run_composed.sh sets them for that
  # arm only, and cx_arm2 exits early without them. A6 (knob 1) must NOT get them.
  local RT=""
  [ "$knob" = "3" ] && RT="yes"
  ( cd "$d" && timeout 21600 nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 \
      RESC_PROBE_PORT=$pp ${RT:+RESC_RETAIN=1 RESC_RETAIN_Q=100} \
      "$BIN/$bin" $ARGS > run.log 2>&1; echo $? > exit.code ) &
}

launch PP0   cx_arm2 3 0     # CONTROL: must reproduce 1,227,505 BIT-IDENTICALLY (guard P1)
launch PP1   cx_arm2 3 1
launch PP2   cx_arm2 3 2
launch PP4   cx_arm2 3 4
launch PP8   cx_arm2 3 8
launch PPA6  cx_arm6 1 0     # opponent, no probes: must reproduce 2,273,856 (guard P6)
wait

echo "=== DONE ==="
for nm in PP0 PP1 PP2 PP4 PP8 PPA6; do
  d=$W/$nm; L=$d/run.log
  printf "PROBEPORT %-5s exit=%-4s cyc=%-9s insn=%-11s %s banner=%s events=%s\n" "$nm" \
    "$([ -f $d/exit.code ] && cat $d/exit.code || echo NONE)" \
    "$(grep -m1 'gpu_sim_cycle =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -m1 'gpu_tot_sim_insn =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -oE 'OUT_CHECKSUM[^ ]* sum=[-0-9.]+' $L 2>/dev/null | head -1)" \
    "$(grep -c '\[RESC_PROBE_PORT\]' $L 2>/dev/null)" \
    "$(grep -oE 'inject_bw_bits=[0-9.]+ events=[0-9]+' $L 2>/dev/null | tail -1)"
done
