#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# ATTRIBUTION make-or-break: arm3 (full RESC) over arm2 (DeNovo safe single
# keep-valid interval) in the LAG>=2 pipeline regime + concurrent-MW regime.
# arm2 = RESC_GEN_RING=2 (DeNovo cur+next), RESC_MW OFF
# arm3 = RESC_GEN_RING=<LAG+1> (>=3), RESC_MW ON
# arm0 = correct global/DSM incumbent (eager drain, G=1), arm1 = unsafe upper bound
# All numbers = real runs. cfg_h20. Distributed suppliers (pipe_chain, fan-in=1,
# NON-strawman: each producer serves exactly one consumer, no injection cliff).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
OUT=ATTRIBUTION_MAKEORBREAK.dat
NC=4; NS=32; TL=4; TO=400
echo "# ATTRIBUTION make-or-break sweep $(date -u +%Y-%m-%dT%H:%M:%SZ)" > $OUT
echo "# cfg_h20, pipe_chain distributed suppliers (fan-in=1, non-strawman), charge ON" >> $OUT
echo "# arm0=G1(eager) arm2=G2(DeNovo cur+next) arm3=G(LAG+1) full-RESC ring" >> $OUT
echo "# LAG arm0_G1 arm2_G2 arm3_Gopt served0 served2 served3 bit0 bit2 bit3 arm3_over_arm2" >> $OUT
for LAG in 1 2 3 4; do
  G3=$((LAG+1))
  o0=$(timeout $TO env RESC_GEN_RING=1 ./pipe_chain $NC $NS $LAG $TL 2>/dev/null)
  o2=$(timeout $TO env RESC_GEN_RING=2 ./pipe_chain $NC $NS $LAG $TL 2>/dev/null)
  o3=$(timeout $TO env RESC_GEN_RING=$G3 ./pipe_chain $NC $NS $LAG $TL 2>/dev/null)
  c0=$(echo "$o0"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+")
  c2=$(echo "$o2"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+")
  c3=$(echo "$o3"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+")
  s0=$(echo "$o0"|grep RESC_GEN_STATS|grep -oE "readshare_served=[0-9]+"|cut -d= -f2)
  s2=$(echo "$o2"|grep RESC_GEN_STATS|grep -oE "readshare_served=[0-9]+"|cut -d= -f2)
  s3=$(echo "$o3"|grep RESC_GEN_STATS|grep -oE "readshare_served=[0-9]+"|cut -d= -f2)
  b0=$(echo "$o0"|grep PIPE_CHAIN|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2)
  b2=$(echo "$o2"|grep PIPE_CHAIN|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2)
  b3=$(echo "$o3"|grep PIPE_CHAIN|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2)
  ratio=$(awk "BEGIN{if($c2>0)printf \"%.3f\", $c3/$c2; else print \"NA\"}")
  echo "LAG=$LAG G3=$G3 arm0=$c0 arm2=$c2 arm3=$c3 served0=$s0 served2=$s2 served3=$s3 bit=$b0/$b2/$b3 arm3_over_arm2=$ratio" | tee -a $OUT
done
echo "MAKEORBREAK_DONE" | tee -a $OUT
