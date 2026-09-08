#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# RESC multi-writer full arm matrix runner.
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
BIN="$1"; BASE="$2"; NC="$3"; RR="$4"; OUT="$5"
TO=250
{
echo "=== ARM1 CORRECT-GLOBAL (no RESC) ==="
timeout $TO ./$BASE $NC $RR 2>/dev/null | grep -iE "MW_REDUCE|MW_STENCIL|RESC_MW_STATS|gpu_tot_sim_cycle ="
echo "=== ARM2 RESC-MULTI-WRITER (RESC_MW=1) ==="
RESC_MW=1 timeout $TO ./$BIN $NC $RR 2>/dev/null | grep -iE "MW_REDUCE|MW_STENCIL|RESC_MW_STATS|gpu_tot_sim_cycle ="
echo "=== ARM3 RESC-OLD-ESCAPE (RESC_MW=0) ==="
RESC_MW=0 timeout $TO ./$BIN $NC $RR 2>/dev/null | grep -iE "MW_REDUCE|MW_STENCIL|RESC_MW_STATS|gpu_tot_sim_cycle ="
echo "=== ARM4 RESC-WHOLE-LINE-LWW (RESC_MW=1 LWW=1) ==="
RESC_MW=1 RESC_MW_LWW=1 timeout $TO ./$BIN $NC $RR 2>/dev/null | grep -iE "MW_REDUCE|MW_STENCIL|RESC_MW_STATS|gpu_tot_sim_cycle ="
} > "$OUT" 2>&1
echo "WROTE $OUT"
