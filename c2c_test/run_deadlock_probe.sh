#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
nvcc -arch=sm_90 resc_e2e.cu -o resc_dl -DBACKEND=2 -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 -DREAD_MODE=1 -DWRITE_MODE=1 -lcudart 2>/dev/null && patchelf --remove-rpath resc_dl 2>/dev/null
echo "built resc_dl"
for nc in 150 200 220 256 300; do
  d=dl_$nc; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
  ( cd $d; timeout 180 ../resc_dl cluster_size=8 repeat=2 num_clusters=$nc avg_elems=128 burst_period=16 reuse_passes=2 > run.log 2>&1; echo "EXIT=$?" >> run.log )
  fin=$(grep -c gpu_tot_sim_cycle $d/run.log 2>/dev/null)
  ex=$(grep "EXIT=" $d/run.log | tail -1)
  lastcyc=$(grep -oE "gpu_sim_cycle = [0-9]+|Cycle [0-9]+|deadlock|DEADLOCK|Stall|SM [0-9]+ .*stuck" $d/run.log 2>/dev/null | tail -2 | tr "\n" " ")
  echo "nc=$nc finished=$fin $ex | $lastcyc"
done
echo DEADLOCK_PROBE_DONE
