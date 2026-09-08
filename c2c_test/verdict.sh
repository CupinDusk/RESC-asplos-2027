#!/bin/bash
# Compute P1-P6 and KILL-C1/C2/C3 from the extracted table. Arithmetic only --
# every verdict traces to measured columns. Unfinished cells are NEVER imputed.
T=${1:?usage: verdict.sh <table.txt>}
awk '
NR==1 { for(i=1;i<=NF;i++) c[$i]=i; next }
{ k=$c["point"]"_"$c["arm"]
  cyc[k]=$c["cycles"]; cps[k]=$c["ctas_per_sm"]; smem[k]=$c["smem_cta_B"]
  l1[k]=$c["l1_KB"]; hr[k]=$c["priv_unm_hr"]; uh[k]=$c["priv_unm_hit"]
  um[k]=$c["priv_unm_miss"]; insn[k]=$c["insn"]; bit[k]=$c["bit"]
  kg[k]=$c["knob_guard"]; ec[k]=$c["exit"]; nm[k]=$c["arm_name"]
  if(!( $c["point"] in seenp )){ seenp[$c["point"]]=1; plist[++np]=$c["point"] }
}
END{
  printf "\n================ PER-POINT VERDICTS ================\n"
  for(p=1;p<=np;p++){ pt=plist[p]
    a2=pt"_A2"; a0=pt"_A0"; a5=pt"_A5"; a6=pt"_A6"; a7=pt"_A7"
    printf "\n--- point %s ---\n", pt
    if(!(a2 in cyc) || cyc[a2]=="NA"){ printf "  RESC (A2) MISSING/UNFINISHED -> point undecidable\n"; continue }
    printf "  %-4s %-22s cycles=%-10s ctas/sm=%-4s smem=%-8s L1KB=%-4s priv_hr=%-10s insn=%-12s bit=%s knob=%s exit=%s\n",
      "A2", nm[a2], cyc[a2], cps[a2], smem[a2], l1[a2], hr[a2], insn[a2], bit[a2], kg[a2], ec[a2]
    split("A0 A5 A6 A7", arms, " ")
    for(i=1;i<=4;i++){ k=pt"_"arms[i]
      if(!(k in cyc) || cyc[k]=="NA"){ printf "  %-4s UNFINISHED (reported as unfinished, never imputed)\n", arms[i]; continue }
      r=cyc[k]/cyc[a2]
      printf "  %-4s %-22s cycles=%-10s ctas/sm=%-4s smem=%-8s L1KB=%-4s priv_hr=%-10s insn=%-12s bit=%s knob=%s exit=%s  vs_A2=%.4fx %s\n",
        arms[i], nm[k], cyc[k], cps[k], smem[k], l1[k], hr[k], insn[k], bit[k], kg[k], ec[k], r,
        (r>1.05 ? "(A2 WINS >5%)" : (r<0.95 ? "(A2 LOSES >5%)" : "(WITHIN 5% = TIE)"))
    }
    # P1 / KILL-C1 hinge on A7
    if((a7 in cyc) && cyc[a7]!="NA"){
      r7=cyc[a7]/cyc[a2]
      printf "  P1/KILL-C1 hinge: A7/A2 = %.4fx -> %s\n", r7,
        (r7>1.05 ? "A7 slower than A2 by >5% : P1 SATISFIED at this point" : "A7 within/below 5% : NO P1 support here")
      printf "  P2 hinge: A7 ctas/sm=%s vs A2 ctas/sm=%s -> %s\n", cps[a7], cps[a2],
        (cps[a7]!="NA" && cps[a2]!="NA" && cps[a7]+0 < cps[a2]+0 ? "STRICTLY LOWER : P2 satisfied" : "NOT strictly lower : P2 unsupported here")
    }
    if((a6 in cyc) && cyc[a6]!="NA" && (a7 in cyc) && cyc[a7]!="NA"){
      printf "  P3 signatures: A6 priv_hr=%s occ=%s | A7 priv_hr=%s occ=%s\n", hr[a6], cps[a6], hr[a7], cps[a7]
    }
  }
  printf "\n================ GLOBAL KILL CONDITIONS ================\n"
  # KILL-C1: A7 within 5% of A2 at EVERY measured point
  n7=0; tie7=0; win7=0
  for(p=1;p<=np;p++){ pt=plist[p]; a2=pt"_A2"; a7=pt"_A7"
    if((a2 in cyc)&&(a7 in cyc)&&cyc[a2]!="NA"&&cyc[a7]!="NA"){ n7++
      r=cyc[a7]/cyc[a2]; if(r<=1.05) tie7++; else win7++ } }
  if(n7==0) printf "KILL-C1: UNDECIDED (no point has both A2 and A7 finished)\n"
  else if(n7<np) printf "KILL-C1: PROVISIONAL -- A7 vs A2 decided at only %d of %d point(s) (%d tie, %d A2-win). NOT declared on partial data.\n", n7, np, tie7, win7
  else if(tie7==n7) printf "KILL-C1: *** FIRED *** A7 within 5%% of A2 at ALL %d measured point(s): the strongest legal software configuration TIES RESC; the composed thesis has NO measured support at these points.\n", n7
  else printf "KILL-C1: NOT fired (A7 slower than A2 by >5%% at %d of %d point(s))\n", win7, n7
  # KILL-C3: no legal arm ever slower than A2
  anyslower=0; tot=0
  for(p=1;p<=np;p++){ pt=plist[p]; a2=pt"_A2"
    split("A0 A5 A6 A7", arms, " ")
    for(i=1;i<=4;i++){ k=pt"_"arms[i]
      if((a2 in cyc)&&(k in cyc)&&cyc[a2]!="NA"&&cyc[k]!="NA"){ tot++
        if(cyc[k]/cyc[a2] > 1.05) anyslower++ } } }
  expected=np*4
  if(tot==0) printf "KILL-C3: UNDECIDED (no comparable pairs finished)\n"
  else if(tot<expected) printf "KILL-C3: PROVISIONAL -- only %d of %d expected arm-point comparisons finished (%d are >5%% slower than A2 so far). NOT declared on partial data.\n", tot, expected, anyslower
  else if(anyslower==0) printf "KILL-C3: *** FIRED *** no legal arm is slower than A2 anywhere -> NULL on the composed thesis.\n"
  else printf "KILL-C3: NOT fired (%d of %d arm-point comparisons are >5%% slower than A2)\n", anyslower, tot
  # KILL-C2: wins only where occupancy differs AND priv hit rate unchanged
  printf "KILL-C2: evaluate from the per-point lines above (needs occupancy differing AND priv_hr unchanged at the winning point).\n"
}' "$T"
