#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# RESC_QUIET equivalence check: knob 2 on selprobe, loud vs quiet.
set +u
source /legacy-research/owner/miniconda3/etc/profile.d/conda.sh
conda activate clustersim-py311
cd /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test
export PATH=$PWD/ptxas_shim:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/legacy-research/owner/cudatest/RESC-c2c-wt/build/lib:$HOME/miniconda3/envs/clustersim-py311/lib:$LD_LIBRARY_PATH
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/qe
mkdir -p $W
for v in loud quiet; do
  d=$W/$v; mkdir -p $d
  cp cfg_h20.config $d/gpgpusim.config
  printf "\n-gpgpu_cluster_barrier_invalidate_l1 2\n" >> $d/gpgpusim.config
  cp /legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/selinv/pc/selprobe $d/
  if [ "$v" = quiet ]; then Q=1; else Q=0; fi
  ( cd $d && timeout 1800 env RESC_QUIET=$Q ./selprobe > run.log 2>&1; echo $? > exit.code ) &
done
wait
echo "QE_DONE"
