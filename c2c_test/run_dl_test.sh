#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh; conda activate clustersim-py311 2>/dev/null
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:/legacy-research/owner/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
export PATH=$PWD/ptxas_shim:$PATH
# Test hypothesis: the "deadlock" at nc>=220 is the [RESC] printf I/O flood, not a protocol cycle.
# Redirect stdout to /dev/null (printf still costs CPU but no terminal/file I/O), give 600s.
# Capture ONLY the final cycle line via stderr-independent check: grep the completion from a tee of gpgpu stats.
for nc in 220 256; do
  d=dltest_$nc; mkdir -p $d; cp cfg_h20.config $d/gpgpusim.config
  ( cd $d
    t0=$(date +%s)
    # keep stdout (has gpu_tot_sim_cycle) but strip [RESC] lines on the fly so I/O is bounded
    timeout 600 ../resc_dl cluster_size=8 repeat=2 num_clusters=$nc avg_elems=128 burst_period=16 reuse_passes=2 2>/dev/null | grep -vE "\[RESC\] (Read|Write) share" > filtered.log
    ec=$?
    t1=$(date +%s)
    cyc=$(grep gpu_tot_sim_cycle filtered.log | tail -1 | grep -oE "[0-9]+" | head -1)
    echo "nc=$nc exit=$ec elapsed=$((t1-t0))s cyc=${cyc:-NONE}"
  )
done
echo DL_TEST_DONE
