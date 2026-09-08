#!/bin/bash
# Historical execution guard; no server is configured here.
printf '%s\n' 'Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.' >&2
exit 64
# Extract the RESC RETENTION sweep into a flat table, computing every guard at
# EVERY point (prereg S6 G1..G8).  Nothing here is asserted from a spot check.
# Dir naming: <arm>_cl<comm_lines>_pv<priv_lines>
W=/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/retain/sweep
cd $W || exit 1
echo "arm CL PV parsed_knob cycles insn ctas cta_limit ctas_per_sm smem_cta sim_static_smem_B sim_dyn_smem_B l1_assoc_postsplit l1_KB_postsplit l1_unified_KB prot_KB ratio l1mr l2mr resfail dram served expired mrk_hit mrk_miss mrk_hr unm_hit unm_miss unm_hr walks prot_seen biased spills spill_rate quota_ovr setfull no_epoch leg_agree leg_differ stamp_unmk exit bit checksum"
for d in */; do
  d=${d%/}
  [ -f "$d/run.log" ] || continue
  arm=$(echo $d | cut -d_ -f1)
  CL=$(echo $d | cut -d_ -f2 | tr -d cl)
  PV=$(echo $d | cut -d_ -f3 | tr -d pv)
  ec=$(cat $d/exit.code 2>/dev/null || echo NA)
  L=$d/run.log
  # the knob the SIMULATOR parsed, not what we wrote (G7)
  pk=$(grep -a "gpgpu_cluster_barrier_invalidate_l1" $L | head -1 | awk '{print $2}')
  cyc=$(grep -a "^gpu_tot_sim_cycle" $L | tail -1 | awk '{print $NF}')
  insn=$(grep -a "^gpu_tot_sim_insn" $L | tail -1 | awk '{print $NF}')
  ctas=$(grep -a "^gpu_tot_issued_cta" $L | tail -1 | awk '{print $NF}')
  # G8: occupancy, recomputed from THIS run's log
  smem=$(grep -a "smem = \|Static SMEM\|shmem =" $L | head -1 | awk '{print $NF}')
  # effective L1 geometry as the simulator configured it
  assoc=$(grep -a "^-gpgpu_cache:dl1 " $L | head -1 | sed -n 's/.*S:\([0-9]*\):\([0-9]*\):\([0-9]*\),.*/\3/p')
  nset=$(grep -a "^-gpgpu_cache:dl1 " $L | head -1 | sed -n 's/.*S:\([0-9]*\):\([0-9]*\):\([0-9]*\),.*/\1/p')
  lsz=$(grep -a "^-gpgpu_cache:dl1 " $L | head -1 | sed -n 's/.*S:\([0-9]*\):\([0-9]*\):\([0-9]*\),.*/\2/p')
  uni=$(grep -a "^-gpgpu_unified_l1d_size" $L | head -1 | awk '{print $2}')
  l1mr=$(grep -a "L1D_total_cache_miss_rate" $L | tail -1 | awk '{print $NF}')
  l2mr=$(grep -a "L2_total_cache_miss_rate" $L | tail -1 | awk '{print $NF}')
  # G2 (KILL-R4): reservation fails MUST match across arms at each point
  resfail=$(grep -a "L1D_total_cache_reservation_fails" $L | tail -1 | awk '{print $NF}')
  dram=$(grep -a "^gpgpu_n_mem_read_global" $L | tail -1 | awk '{print $NF}')
  gs=$(grep -a "RESC_GEN_STATS" $L | tail -1)
  served=$(echo "$gs" | sed -n 's/.*readshare_served=\([0-9]*\).*/\1/p')
  expired=$(echo "$gs" | sed -n 's/.*readshare_expired=\([0-9]*\).*/\1/p')
  sp=$(grep -a "RESC_L1SPLIT_STATS" $L | tail -1)
  mh=$(echo "$sp" | sed -n 's/.*[^n]marked_rd_hit=\([0-9]*\).*/\1/p')
  mm=$(echo "$sp" | sed -n 's/.*[^n]marked_rd_miss=\([0-9]*\).*/\1/p')
  mhr=$(echo "$sp" | sed -n 's/.*[^n]marked_hitrate=\([0-9.]*\).*/\1/p')
  uh=$(echo "$sp" | sed -n 's/.*unmarked_rd_hit=\([0-9]*\).*/\1/p')
  um=$(echo "$sp" | sed -n 's/.*unmarked_rd_miss=\([0-9]*\).*/\1/p')
  uhr=$(echo "$sp" | sed -n 's/.*unmarked_hitrate=\([0-9.]*\).*/\1/p')
  rt=$(grep -a "RESC_RETAIN_STATS" $L | tail -1)
  walks=$(echo "$rt" | sed -n 's/.*victim_walks=\([0-9]*\).*/\1/p')
  pseen=$(echo "$rt" | sed -n 's/.*prot_candidates_seen=\([0-9]*\).*/\1/p')
  biased=$(echo "$rt" | sed -n 's/.*biased=\([0-9]*\).*/\1/p')
  spills=$(echo "$rt" | sed -n 's/.*spills=\([0-9]*\).*/\1/p')
  srate=$(echo "$rt" | sed -n 's/.*spill_rate=\([0-9.]*\).*/\1/p')
  qovr=$(echo "$rt" | sed -n 's/.*quota_override=\([0-9]*\).*/\1/p')
  sfull=$(echo "$rt" | sed -n 's/.*set_full=\([0-9]*\).*/\1/p')
  noep=$(echo "$rt" | sed -n 's/.*no_epoch=\([0-9]*\).*/\1/p')
  lagr=$(echo "$rt" | sed -n 's/.*legacy_agree=\([0-9]*\).*/\1/p')
  ldif=$(echo "$rt" | sed -n 's/.*legacy_differ=\([0-9]*\).*/\1/p')
  sunm=$(echo "$rt" | sed -n 's/.*stamp_unmarked=\([0-9]*\).*/\1/p')
  mf=$(grep -a "^MARKFRAC" $L | tail -1)
  bit=$(echo "$mf" | sed -n 's/.*BITIDENTICAL=\([0-9]*\).*/\1/p')
  ck=$(echo "$mf" | sed -n 's/.*OUT_CHECKSUM=\([0-9]*\).*/\1/p')
  # G8 occupancy, COMPUTED from this run's own numbers -- never asserted.
  # CAREFUL: the simulator's "CTA/core = N" line prints at shader.cc:3565, which
  # is BEFORE the k.num_blocks() clamp at :3576-3578, so N is the RESOURCE LIMIT
  # (what would fit), NOT the CTAs/SM actually resident.  Record both.
  ctalim=$(grep -a "CTA/core = " $L | head -1 | sed -n 's/.*CTA\/core = \([0-9]*\).*/\1/p')
  nsm=$(grep -a "^-gpgpu_n_clusters" $L | head -1 | awk '{print $2}')
  # INDEPENDENT SMEM readback from the simulator's own launch banner
  # (libcuda/cuda_runtime_api.cc:1040-1042), not from the benchmark.
  sstat=$(grep -a "Static Shared Memory Per Block" $L | head -1 | awk '{print $(NF-1)}')
  sdyn=$(grep -a "Dynamic Shared Memory Per Block" $L | head -1 | awk '{print $(NF-1)}')
  cps=NA
  if [ -n "$ctas" ] && [ -n "$nsm" ] && [ "$nsm" != "0" ]; then
    cps=$(awk -v b=$ctas -v s=$nsm -v lim=${ctalim:-0} 'BEGIN{
      r=int(b/s); if (b%s) r++; if (lim>0 && r>lim) r=lim; print r}')
  fi
  # L1 KB and protected-footprint KB, COMPUTED (Axis A ratio).
  # AUTHORITATIVE post-split L1 size: the "-gpgpu_cache:dl1" echo shows the BASE
  # assoc (64) BEFORE the adaptive split, so it must not be used here.
  # shader.cc:3635-3636 prints the size AFTER set_assoc() as
  # "GPGPU-Sim: Reconfigure L1 cache to <N>KB".  cfg_h20_tinyL1/smallL1 are proof
  # that unified size and real L1 can differ (unified 4 or 16 against a 32 KB
  # base gives assoc 0 by integer division, then deadlock).
  l1kb=$(grep -a "Reconfigure L1 cache to" $L | head -1 | sed -n 's/.*to \([0-9]*\)KB.*/\1/p')
  ratio=NA; pkb=NA; assoc_post=NA
  if [ -n "$l1kb" ] && [ -n "$nset" ] && [ -n "$lsz" ] && [ "$nset" != "0" ]; then
    assoc_post=$(awk -v k=$l1kb -v n=$nset -v l=$lsz 'BEGIN{printf "%.0f", k*1024/(n*l)}')
  fi
  if [ -n "$l1kb" ]; then
    # protected footprint per SM = W(=G=3, or 1 for R3) * 2 tiles * CL * 128B
    Wn=3; [ "$arm" = "R3" ] && Wn=1
    pkb=$(awk -v w=$Wn -v cl=$CL 'BEGIN{printf "%.3f", w*2*cl*128/1024}')
    ratio=$(awk -v p=$pkb -v l=$l1kb 'BEGIN{if(l>0) printf "%.4f", p/l; else print "NA"}')
  fi
  echo "$arm $CL $PV ${pk:-NA} ${cyc:-NA} ${insn:-NA} ${ctas:-NA} ${ctalim:-NA} ${cps:-NA} ${smem:-0} ${sstat:-NA} ${sdyn:-NA} ${assoc_post:-NA} ${l1kb:-NA} ${uni:-NA} ${pkb:-NA} ${ratio:-NA} ${l1mr:-NA} ${l2mr:-NA} ${resfail:-NA} ${dram:-NA} ${served:-NA} ${expired:-NA} ${mh:-NA} ${mm:-NA} ${mhr:-NA} ${uh:-NA} ${um:-NA} ${uhr:-NA} ${walks:-NA} ${pseen:-NA} ${biased:-NA} ${spills:-NA} ${srate:-NA} ${qovr:-NA} ${sfull:-NA} ${noep:-NA} ${lagr:-NA} ${ldif:-NA} ${sunm:-NA} $ec ${bit:-NA} ${ck:-NA}"
done | sort -k3,3n -k2,2n -k1,1
