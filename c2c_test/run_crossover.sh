#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source ~/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
cp cfg_h20.config gpgpusim.config
NC=${NC:-4}; NS=${NS:-32}; LAG=${LAG:-2}; TL=${TL:-4}; CI=${CI:-0}; TO=${TO:-600}
echo "### CROSSOVER pipe_fanout NC=$NC NS=$NS LAG=$LAG TL=$TL compute_iters=$CI (charge ON)"
echo "# nprod fan G1_eager G3_overlap ratio(ov/eager) served1 served3 bit"
for P in "$@"; do
  o1=$(timeout $TO env RESC_GEN_RING=1 ./pipe_fanout $NC $NS $LAG $TL $P $CI 2>/dev/null)
  o3=$(timeout $TO env RESC_GEN_RING=3 ./pipe_fanout $NC $NS $LAG $TL $P $CI 2>/dev/null)
  c1=$(echo "$o1"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+")
  c3=$(echo "$o3"|grep "gpu_tot_sim_cycle ="|tail -1|grep -oE "[0-9]+")
  fan=$(echo "$o1"|grep PIPE_FANOUT|grep -oE "fan=[0-9]+"|cut -d= -f2)
  s1=$(echo "$o1"|grep RESC_GEN_STATS|grep -oE "readshare_served=[0-9]+"|cut -d= -f2)
  s3=$(echo "$o3"|grep RESC_GEN_STATS|grep -oE "readshare_served=[0-9]+"|cut -d= -f2)
  b1=$(echo "$o1"|grep PIPE_FANOUT|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2)
  b3=$(echo "$o3"|grep PIPE_FANOUT|grep -oE "BITIDENTICAL=[0-9]+"|cut -d= -f2)
  ratio=$(awk "BEGIN{if($c1>0)printf \"%.3f\", $c3/$c1; else print \"NA\"}")
  echo "P=$P fan=$fan eager=$c1 overlap=$c3 ratio=$ratio served1=$s1 served3=$s3 bit=$b1/$b3"
done
echo "CROSSOVER_DONE"
