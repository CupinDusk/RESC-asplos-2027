#!/usr/bin/env python3
"""Recompute simulator-vs-real-H20 error for each fidelity configuration,
using the SAME instrument and the SAME real-H20 reference values that produced
resc-regime-paper/data/validation_crossover.dat."""
# Historical execution guard; no server is configured here.
raise SystemExit('Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.')
import re, sys, os, json

BASE = "/legacy-research/owner/cudatest/RESC-c2c-wt/c2c_test/fidfix/xval"

# Real-H20 reference (from data/validation_crossover.dat, unchanged)
REAL = {
 (2,64):(689.0,836.5),  (2,256):(1995.5,2803.0), (2,512):(3777.0,5233.5), (2,1024):(7299.5,10179.0),
 (4,64):(705.5,822.5),  (4,256):(2068.5,2842.5), (4,512):(3848.5,5383.5), (4,1024):(7445.5,10606.5),
 (8,64):(706.0,814.0),  (8,256):(2065.0,2824.5), (8,512):(3842.0,5363.5), (8,1024):(7427.5,10500.0),
}
# Published sim values (pre-fix) for the reproduction check
PUB = {
 (2,64):(542.0,711.0),  (2,256):(2048.0,2721.0), (2,512):(4056.0,5401.0), (2,1024):(8072.0,10761.0),
 (4,64):(599.0,724.0),  (4,256):(2219.0,2770.0), (4,512):(4369.5,5498.0), (4,1024):(8689.5,10954.0),
 (8,64):(694.0,750.0),  (8,256):(2656.0,2868.0), (8,512):(5272.0,5692.0), (8,1024):(10504.0,11340.0),
}

pat = re.compile(r"DSM=([\d.]+)±[\d.]+ cyc/read\s+GLOBAL=([\d.]+)±")

def read(cfg, cs, nf):
    p = os.path.join(BASE, f"{cfg}_c{cs}_nf{nf}", "run.log")
    if not os.path.exists(p): return None
    txt = open(p, errors="ignore").read()
    m = pat.findall(txt)
    if not m: return None
    d, g = m[-1]
    return float(d), float(g)

cfgs = ["base","inv","l2","both"]
rows = []
print(f"{'cfg':<6}{'cs':<4}{'nf':<6}{'simDSM':>9}{'simGLB':>9}{'errDSM%':>9}{'errGLB%':>9}")
summary = {}
for cfg in cfgs:
    eD, eG, n = [], [], 0
    for cs in (2,4,8):
        for nf in (64,256,512,1024):
            r = read(cfg, cs, nf)
            if r is None: continue
            sd, sg = r
            rd, rg = REAL[(cs,nf)]
            ed = abs(sd-rd)/rd*100.0
            eg = abs(sg-rg)/rg*100.0
            eD.append(ed); eG.append(eg); n += 1
            rows.append((cfg,cs,nf,sd,sg,ed,eg))
            print(f"{cfg:<6}{cs:<4}{nf:<6}{sd:>9.1f}{sg:>9.1f}{ed:>9.1f}{eg:>9.1f}")
    if n:
        summary[cfg] = dict(n=n,
            dsm_mean=sum(eD)/len(eD), dsm_max=max(eD),
            glb_mean=sum(eG)/len(eG), glb_max=max(eG),
            all_mean=(sum(eD)+sum(eG))/(len(eD)+len(eG)))
print()
print(f"{'cfg':<6}{'n':>4}{'DSMmean':>9}{'DSMmax':>9}{'GLBmean':>9}{'GLBmax':>9}{'ALLmean':>9}")
for cfg in cfgs:
    if cfg in summary:
        s = summary[cfg]
        print(f"{cfg:<6}{s['n']:>4}{s['dsm_mean']:>9.2f}{s['dsm_max']:>9.2f}{s['glb_mean']:>9.2f}{s['glb_max']:>9.2f}{s['all_mean']:>9.2f}")

# reproduction check: does 'base' reproduce the published pre-fix table exactly?
print()
bad = 0
for cs in (2,4,8):
    for nf in (64,256,512,1024):
        r = read("base", cs, nf)
        if r is None: continue
        if (abs(r[0]-PUB[(cs,nf)][0])>1e-6) or (abs(r[1]-PUB[(cs,nf)][1])>1e-6):
            print(f"REPRO MISMATCH c{cs} nf{nf}: got {r} published {PUB[(cs,nf)]}")
            bad += 1
print(f"REPRODUCTION_CHECK: {'PASS (base == published pre-fix table, bit-exact)' if bad==0 else f'FAIL ({bad} cells differ)'}")
json.dump(summary, open("/tmp/xval_summary.json","w"), indent=1)
