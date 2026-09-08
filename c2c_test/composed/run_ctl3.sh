#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# CTL3_A0_FIT192: the separating control, derived from maxunion.cu:247 rather than assumed.
# At comm_lines=192: small=24,576 B and big=159,744 B -- BOTH fit the 212,992 B arena.
# Decision frozen in bigcomm-control-void-2-and-derived-size-constraint.md
#   sha256 06fa7410c74b5c9d1d4a020d8860b1df682f05730818ac05d11d6aee13920022
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
set -u
T=/legacy-research/owner/cudatest/RESC-c2c-wt
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$T/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:${LD_LIBRARY_PATH:-}
W=$T/c2c_test/composed/sweep_bigcomm
BINDIR=$T/c2c_test/composed/bin/v124

# 1. Retire the second mis-designed control by name (keep the data, never delete).
old=$W/CTL2_A0_FIT
new=$W/VOID2_misdesigned_CTL_big_payload_851968_over_arena
if [ -d "$old" ] && [ ! -d "$new" ]; then
  for p in $(pgrep -u hyren -f "cx_arm"); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$old" ] && { echo "REFUSE: $old still has live pid $p"; exit 1; }
  done
  mv "$old" "$new" && echo "RENAMED CTL2_A0_FIT -> $(basename $new)"
fi

# 2. Launch the separating control.
nm=CTL3_A0_FIT192
d=$W/$nm
ARGS="2 32 4 192 256"
if [ -f "$d/exit.code" ] && [ "$(cat "$d/exit.code")" = "0" ]; then echo "SKIP $nm complete"; exit 0; fi
for p in $(pgrep -u hyren -f "cx_arm"); do
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$d" ] && { echo "IN-FLIGHT $nm pid=$p"; exit 0; }
done
mkdir -p "$d"
cp "$W/BIG_A0/gpgpusim.config" "$d/gpgpusim.config"
grep -q "dl2 S:960" "$d/gpgpusim.config" || { echo "FATAL no S:960"; exit 1; }
rm -f "$d/exit.code"
(
  cd "$d" || exit 1
  RESC_QUIET=1 RESC_GEN_RING=3 RESC_MW=1 \
  timeout 21600 nice -n 10 "$BINDIR/cx_arm0" $ARGS > run.log 2>&1
  echo $? > exit.code
) &
echo "LAUNCHED $nm argv='$ARGS' (small=24576 big=159744, both fit 212992)"
