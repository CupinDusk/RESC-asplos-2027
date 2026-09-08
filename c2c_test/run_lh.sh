#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Does latency hiding shrink the granularity ratio? -- the r33 gpu-arch objection.
# Prereg: latency-hiding-interaction-prereg.sha256
# Controls: LH4_RESC must be 131,665 and LH4_A6 must be 136,210 (stock -gpgpu_num_sched_per_core 4).
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
BIN=$T/c2c_test/attn_a6/bin
SRC_RESC=$T/c2c_test/attn_a6/AT_RESC/gpgpusim.config
SRC_A6=$T/c2c_test/attn_a6/AT_A6/gpgpusim.config
W=$T/c2c_test/attn_a6/sweep_lathide
ARGS="4 8 8"
MAXJOBS=3
WANT=fcfd697021e8033eb77b8f64259b6382

for f in "$SRC_RESC" "$SRC_A6"; do [ -f "$f" ] || { echo "FATAL missing $f"; exit 1; }; done
GOT=$(md5sum "$BIN/dec_resc" | awk '{print $1}')
[ "$GOT" = "$WANT" ] || { echo "FATAL binary drift $GOT"; exit 1; }
mkdir -p "$W"
echo "### .so md5: $(md5sum $T/build/lib/libcudart.so.12 | awk '{print $1}')  dec_resc md5: $GOT"

running(){ ps -u hyren -o comm --no-headers | grep -c '^dec_resc'; }
throttle(){ while [ "$(running)" -ge $MAXJOBS ]; do sleep 20; done; }

# $1=nsched $2=arm(RESC|A6) $3=knob $4=srccfg
launch(){
  local ns="$1"
  local arm="$2"
  local knob="$3"
  local src="$4"
  local d="$W/LH${ns}_${arm}"
  if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP LH${ns}_${arm}"; return; fi
  mkdir -p "$d"
  sed "s/^-gpgpu_num_sched_per_core .*/-gpgpu_num_sched_per_core $ns/" "$src" > "$d/gpgpusim.config"
  grep -q "^-gpgpu_num_sched_per_core $ns\$" "$d/gpgpusim.config" || { echo "FATAL nsched LH${ns}_${arm}"; return; }
  grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL S:960 LH${ns}_${arm}"; return; }
  grep -q "^-gpgpu_cluster_barrier_invalidate_l1 $knob\$" "$d/gpgpusim.config" || { echo "FATAL knob LH${ns}_${arm}"; return; }
  rm -f "$d/exit.code" "$d/run.log"
  throttle
  echo "LAUNCH LH${ns}_${arm} nsched=$ns knob=$knob"
  ( cd "$d" && timeout 21600 nice -n 10 env RESC_QUIET=1 RESC_GEN_RING=8 RESC_MW=1 \
      "$BIN/dec_resc" $ARGS > run.log 2>&1; echo $? > exit.code ) &
}

for ns in 4 2 1; do
  launch $ns RESC 3 "$SRC_RESC"
  launch $ns A6   1 "$SRC_A6"
done
wait

echo "=== DONE ==="
for ns in 4 2 1; do for arm in RESC A6; do
  d=$W/LH${ns}_${arm}; L=$d/run.log
  printf "LH%-2s %-5s exit=%-4s cyc=%-8s insn=%-11s %s served=%s nsched=%s\n" "$ns" "$arm" \
    "$([ -f $d/exit.code ] && cat $d/exit.code || echo NONE)" \
    "$(grep -m1 'gpu_sim_cycle =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -m1 'gpu_tot_sim_insn =' $L 2>/dev/null | awk '{print $3}')" \
    "$(grep -oE 'OUT_CHECKSUM sum=[-0-9.]+' $L 2>/dev/null | head -1)" \
    "$(grep -oE 'readshare_served=[0-9]+' $L 2>/dev/null | tail -1 | cut -d= -f2)" \
    "$(grep -oE '^-gpgpu_num_sched_per_core [0-9]+' $d/gpgpusim.config 2>/dev/null | awk '{print $2}')"
done; done
