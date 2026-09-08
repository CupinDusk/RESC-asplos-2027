#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# W1 sensitivity sweep: scale sm_2_sm_network_latency 0.5-2x (98-390 cyc) around
# the 195 baseline. Confirm the two headline wins survive across the band:
#  - legality: RESC vs correct-cg at equal occupancy, reuse=8, 8KB payload
#  - occupancy: RESC vs DSM under SMEM pressure (proxy: comm-only at cluster8)
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
# build cg(correct baseline) + resc, no compute tile (pure legality axis)
nvcc -arch=sm_90 resc_e2e.cu -o cg_s -DBACKEND=0 -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 -lcudart 2>/dev/null && patchelf --remove-rpath cg_s 2>/dev/null
nvcc -arch=sm_90 resc_e2e.cu -o rc_s -DBACKEND=2 -DRESERVE_ELEMS=8192 -DSTRIDE_ELEMS=16384 -DTILE_KB=0 -DCOMPUTE_ITERS=0 -DREAD_MODE=1 -DWRITE_MODE=1 -lcudart 2>/dev/null && patchelf --remove-rpath rc_s 2>/dev/null
echo "# W1 forwarding-latency sensitivity: legality (RESC vs correct-cg), 8KB payload reuse=8, equal occupancy"
echo "# lat_cyc cg_cyc resc_cyc resc_speedup"
# cg is latency-invariant (bypasses L1, no C2C); compute it once
cg=$(cd /tmp; d=cgbase; mkdir -p $d; cd $d; cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/cfg_h20.config gpgpusim.config; timeout 400 /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/cg_s cluster_size=2 repeat=2 num_clusters=40 avg_elems=2048 burst_period=16 reuse_passes=8 2>/dev/null | grep -vE "\[RESC\]" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
for lat in 98 146 195 244 293 390; do
  d=sens_$lat; mkdir -p $d
  sed "s/-sm_2_sm_network_latency 195/-sm_2_sm_network_latency $lat/" cfg_h20.config > $d/gpgpusim.config
  rc=$(cd $d; timeout 400 ../rc_s cluster_size=2 repeat=2 num_clusters=40 avg_elems=2048 burst_period=16 reuse_passes=8 2>/dev/null | grep -vE "\[RESC\]" | grep gpu_tot_sim_cycle | sed -E "s/.*= *//" | grep -oE "[0-9]+" | head -1)
  sp=$(python3 -c "print(f\"{$cg/$rc:.3f}\") " 2>/dev/null)
  echo "$lat $cg ${rc:-NA} ${sp:-NA}"
done
echo SENSITIVITY_DONE
