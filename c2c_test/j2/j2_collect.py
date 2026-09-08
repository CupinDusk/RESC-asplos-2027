#!/usr/bin/env python3
"""Collect Job-2 head-to-head results into a .dat file with 18.9% intervals."""
# Historical execution guard; no server is configured here.
raise SystemExit('Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.')
import os, re, sys, glob

BASE = "/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/j2"
ERR = 1.189  # 18.9% held-out validation error at cluster-8
ARMS = {0:"cg_relaxed",1:"dsm",2:"resc_ub",3:"ca_acq",4:"cg_acq",5:"hybrid",6:"resc_faith"}

def parse(d):
    lg = os.path.join(d,"run.log")
    if not os.path.exists(lg): return None
    txt = open(lg,errors="ignore").read()
    if "J2 " not in txt: return None
    def g(pat):
        m = re.search(pat, txt, re.M)
        return m.group(1) if m else None
    cyc = g(r"^gpu_sim_cycle = (\d+)")
    ck  = g(r"CHECKSUM=([0-9.]+)")
    hit = g(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\] = (\d+)")
    mis = g(r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\] = (\d+)")
    l2  = g(r"L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\] = (\d+)")
    if cyc is None: return None
    return dict(cyc=int(cyc), ck=ck, hit=int(hit or 0), miss=int(mis or 0), l2=int(l2 or 0))

rows={}
for d in sorted(glob.glob(os.path.join(BASE,"run_a*"))):
    m = re.match(r"run_a(\d+)_cs(\d+)_t(\d+)_r(\d+)$", os.path.basename(d))
    if not m: continue
    arm,cs,t,r = map(int, m.groups())
    p = parse(d)
    if p: rows[(cs,t,r,arm)] = p

pts = sorted({(cs,t,r) for (cs,t,r,a) in rows})
print("# Job 2: whole-kernel head-to-head, RESC vs the legal ld.ca + cluster-barrier-acquire program")
print("# Machine legacy-user@legacy-host.invalid tree /legacy-research/owner/cudatest/RESC-c2c-wt (HEAD d3cc7ad7)")
print("# Build UNMODIFIED, libcudart.so.12 md5 041fe31edbf4da088d25af546071d586")
print("# Sanity gate ./simxl_c8 256 -> DSM=2656.0 GLOBAL=2868.0 ratio=0.926 (PASS)")
print("# Pre-registration sim-headtohead-prereg.md sha256 6eb18222fa54d42f924c3f06d4f7e0dab7b1aac575edb17bb9b412d06c811ceb")
print("# Metric = whole-kernel gpu_sim_cycle. EPOCHS=8 NCLUSTERS=2 TILE_PASSES=4 hotset=4KB")
print("# arms: 0 cg_relaxed | 1 dsm | 2 resc_ub(UPPER BOUND) | 3 ca_acq(NEW) | 4 cg_acq | 5 hybrid | 6 resc_faith")
print("# ratios vs ca_acq(arm3); interval = [r/1.189, r*1.189] (18.9% held-out error, cluster-8)")
print("#")
print("# cs tile_kb reuse arm name        cycles   l1_miss  l1_hit    l2_reads  checksum      ratio_vs_ca_acq  lo     hi")
for (cs,t,r) in pts:
    base = rows.get((cs,t,r,3))
    for a in sorted(ARMS):
        p = rows.get((cs,t,r,a))
        if not p: continue
        if base:
            rat = base["cyc"]/p["cyc"]
            lo, hi = rat/ERR, rat*ERR
            rs = f"{rat:.3f}  {lo:.3f}  {hi:.3f}"
        else:
            rs = "NA     NA     NA"
        print(f"{cs:3d} {t:7d} {r:5d} {a:3d} {ARMS[a]:<11s} {p['cyc']:8d} {p['miss']:8d} {p['hit']:9d} {p['l2']:9d} {p['ck']:<13s} {rs}")
