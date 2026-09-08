#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# ATTRIBUTION make-or-break, CONCURRENT-MULTI-WRITER regime.
# Reproduces the 4 arms on the mw_reduce concurrent-writer workload (8 writers/
# cluster). Attribution arm naming (differs from multiwriter.dat legacy naming):
#   arm0 = correct-global baseline  = *_base build (no RESC hook)
#   arm2 = RESC safe single keep-valid interval, NO multi-writer = RESC_MW=0
#          (this is the DeNovo-equivalent: correct for single-writer, but for
#           CONCURRENT writers it uses the old whole-line escape)
#   arm3 = full RESC = RESC_MW=1 (per-sector merge Path A / owner-token Path B)
#   arm4 = whole-line-LWW ablation = RESC_MW=1 RESC_MW_LWW=1
# Path A = false sharing (FORM=1, mw_reduce_fs), Path B = atomics (FORM=0, atomic).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
OUT=ATTRIBUTION_MULTIWRITER.dat
TO=400
NC=8
getcyc(){ echo "$1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+"; }
getbit(){ echo "$1"|grep MW_REDUCE|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2; }
echo "# ATTRIBUTION multi-writer make-or-break $(date -u +%Y-%m-%dT%H:%M:%SZ)" > $OUT
echo "# cfg_h20 cluster=8 (8 concurrent writers). SIM IS TAG-ONLY => in-sim bit" >> $OUT
echo "# always identical; correctness of concurrent MW is established OFF-SIM" >> $OUT
echo "# (correctness/modelcheck_output.txt, resc_mw_harness). Here: timing only." >> $OUT
echo "# path arm cycles bit arm3_over_arm2" >> $OUT

# ---- PATH A false sharing (rounds=5) ----
RR=5
a0=$(timeout $TO ./mw_reduce_fs_base $NC $RR 2>/dev/null)
a2=$(timeout $TO env RESC_MW=0 ./mw_reduce_fs $NC $RR 2>/dev/null)
a3=$(timeout $TO env RESC_MW=1 ./mw_reduce_fs $NC $RR 2>/dev/null)
a4=$(timeout $TO env RESC_MW=1 RESC_MW_LWW=1 ./mw_reduce_fs $NC $RR 2>/dev/null)
c0=$(getcyc "$a0"); c2=$(getcyc "$a2"); c3=$(getcyc "$a3"); c4=$(getcyc "$a4")
r=$(awk "BEGIN{if($c2>0)printf \"%.3f\",$c3/$c2; else print \"NA\"}")
echo "A arm0_correct_global $c0 $(getbit "$a0") -" | tee -a $OUT
echo "A arm2_RESC_noMW(old-escape) $c2 $(getbit "$a2") -" | tee -a $OUT
echo "A arm3_RESC_MW(per-sector-merge) $c3 $(getbit "$a3") $r" | tee -a $OUT
echo "A arm4_wholeLine_LWW_ablation $c4 $(getbit "$a4") -" | tee -a $OUT

# ---- PATH B atomics (rounds=1 due to pre-existing coalescer int64 bug) ----
RR=1
b0=$(timeout $TO ./mw_reduce_atomic_base $NC $RR 2>/dev/null)
b2=$(timeout $TO env RESC_MW=0 ./mw_reduce_atomic $NC $RR 2>/dev/null)
b3=$(timeout $TO env RESC_MW=1 ./mw_reduce_atomic $NC $RR 2>/dev/null)
c0=$(getcyc "$b0"); c2=$(getcyc "$b2"); c3=$(getcyc "$b3")
r=$(awk "BEGIN{if($c2>0)printf \"%.3f\",$c3/$c2; else print \"NA\"}")
echo "B arm0_correct_global $c0 $(getbit "$b0") -" | tee -a $OUT
echo "B arm2_RESC_noMW(old-escape) $c2 $(getbit "$b2") -" | tee -a $OUT
echo "B arm3_RESC_MW(owner-token) $c3 $(getbit "$b3") $r" | tee -a $OUT
echo "MW_ATTRIB_DONE" | tee -a $OUT
