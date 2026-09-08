#!/usr/bin/env python3
# Historical execution guard; no server is configured here.
raise SystemExit('Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.')
# RESC EPOCH-SCOPED RETENTION BIAS -- patch script (v4).
#
# Applies to /legacy-research/owner/cudatest/RESC-c2c-wt/src/gpgpu-sim/ (NOT ClusterSim):
#   gpu-cache.cc : (A) knob + protection predicate + quota + counters
#                  (B) the retention decision at the victim-selection site
#   gpu-sim.cc   : (C) a stats print block beside the existing RESC blocks
# Every edit is anchored on exact existing text and fails loudly if not found.
#
# v4 vs v3: the second pass now initialises its age accumulators to (unsigned)-1,
# exactly as the legacy walk does, so the legacy_agree guard is exact.
#
# v3 vs v2: protection is tested STAMP-FIRST.  resc_stamp_gen() is called only
# under should_set_cluster_state_for_write() (gpu-cache.cc:2711-2715), so a
# stamped line is necessarily a MARKED line; testing the small per-cluster stamp
# maps first turns the predicate into a couple of hash lookups instead of a scan
# of every registry bucket.  Registry membership is then confirmed on the
# positive path only, and any disagreement is COUNTED (must be 0) rather than
# assumed.
#
# v2 vs v1: the protection test is NOT put inside the per-way victim loop.
# probe() walks m_config.m_assoc = 512 ways on EVERY access, so a lookup per way
# would multiply the hottest loop in the simulator.  The legacy walk is left
# BYTE-UNCHANGED and a second pass over the same set runs ONLY when the walk has
# already decided it must evict a valid line (a capacity eviction).  The
# SEMANTICS are identical to the one-pass form hardware would use; the second
# pass selects an index and charges no simulated cycles.
import sys, io

ROOT = "/legacy-research/owner/cudatest/RESC-c2c-wt/src/gpgpu-sim/"

def rd(p):
    with io.open(p, encoding="utf-8") as f: return f.read()
def wr(p, s):
    with io.open(p, "w", encoding="utf-8") as f: f.write(s)
def sub1(s, old, new, tag):
    n = s.count(old)
    if n != 1:
        sys.exit("ANCHOR FAIL [%s]: found %d occurrences, expected 1" % (tag, n))
    return s.replace(old, new, 1)

# ---------------------------------------------------------------- A
A_ANCHOR = "// 用于识别需要设置cluster-state的目标地址（向后兼容，通过环境变量设置）"

A_NEW = r'''// ===========================================================================
// RESC EPOCH-SCOPED RETENTION BIAS (2026-09-04).  DEFAULT OFF.
//
// THE PROBLEM.  Nothing so far keeps a marked communication line resident in
// L1 for the duration of the communication.  A marked line competes for ways
// on equal terms with everything else -- the victim walk in tag_array::probe
// classifies candidates only by dirtiness and age -- so an unrelated private
// read can evict the producer's copy before the consumer asks for it; the
// read-share then fails and the consumer pays L2.  RESC's own residency is
// unmanaged.  This is the untested self-destruct condition the 2026-08-27
// scenario search named.
//
// WHY NOT A PIN.  Pinning ways for marked data re-creates DSM's defect: a
// FIXED reservation, taken whether or not it is used, sized for the worst
// case, and capable of FAILING (no way left to allocate => RESERVATION_FAIL).
// The whole structural argument for RESC over DSM is that its cost is
// VARIABLE -- per line, only for lines actually resident, released
// automatically when the epoch ends.  A pin forfeits that argument.
//
// THE MECHANISM (a bias, three properties, each structural not incidental).
//  1. PREFERENCE, NOT RESERVATION.  Among the lines the legacy policy already
//     considers EVICTABLE, prefer to evict an unprotected one.  Protection
//     never removes a line from the eligible set, so `all_reserved` -- and
//     therefore RESERVATION_FAIL -- is bit-unchanged.  Retention cannot fail
//     and cannot deadlock; it can only reorder which of several evictable
//     lines is evicted.
//  2. BOUNDED BY A PER-SET QUOTA.  If protected lines could take every way the
//     bias WOULD be a reservation in disguise (one unprotected way thrashing
//     forever).  So the bias applies only while the protected population of
//     the set is <= Q ways, Q = assoc * RESC_RETAIN_Q / 100, clamped to
//     [0, assoc-1].  Past Q the protected lines lose the bias and are evicted
//     like anything else -- the SPILL.  Because this L1 is WRITE_THROUGH
//     (-gpgpu_cache:dl1 S:4:128:64,L:T:..., 'T' => WRITE_THROUGH in
//     gpu-cache.h), L2 already holds the authoritative value, so a spilled
//     marked line costs LATENCY (the consumer misses to L2) and never
//     correctness.
//  3. SELF-RELEASING, O(1), NO WALK.  Protection is not a sticky bit that
//     something must clear.  A line is protected iff it carries a producing
//     generation stamp inside its cluster's live epoch window -- the SAME
//     per-line stamp and the SAME per-cluster epoch counter the generation
//     ring already maintains (resc_stamp_gen / resc_gen_live above).  When
//     cluster.sync advances the epoch counter, every line whose stamp has
//     fallen out of the window stops being protected simultaneously, with no
//     walk, no clearing, and no way to leak protection into the next epoch.
//     Epoch release is a comparator, not an operation.
//
// WHAT IS PROTECTED, PRECISELY.  Producer-stamped marked lines: exactly the
// copies a peer's read-share can be served from.  A marked line that this L1
// never produced (a consumer's own fetched copy) is NOT protected -- it has no
// producing generation, and protecting it could not help any handoff.  This is
// the narrow reading, and it is the conservative one: fewer protected lines
// means less benefit claimed and less pressure on unmarked data.
//
// HARDWARE COST (what the paper must claim).  Per line: the E-bit generation
// stamp the ring already requires (E = ceil(log2 G) + 1) plus a 1-bit
// "stamped/marked" valid flag.  Per way, in the tag-probe stage and in parallel
// with the existing tag compare: one E-bit modular window compare ANDed with
// that flag.  Per set access: one extra (age,index) candidate register pair and
// a log2(assoc)-bit population count over the way vector -- the popcount is
// strictly narrower than the assoc-way 64-bit MINIMUM reduction LRU already
// performs in the same stage, and it disappears entirely at Q = 100%.  Nothing
// is added to the hit path; no new array is read.
//
// MODELLING CONVENIENCES, DISCLOSED.  (a) Hardware classifies ways in the one
// walk it already does; the model runs a second pass over the same set, only on
// capacity evictions, purely to keep the hot loop's cost unchanged.  The
// selection is identical either way and the second pass charges no cycles.
// (b) The model recovers stamp/marked state from the existing global maps
// instead of per-line bits; hardware needs only the bits above, and needs no
// cluster key because an L1 serves one cluster at a time.  (c) Wrap-around is a
// modular compare over a window G <= 2^(E-1).
//
// DEPENDENCY, STATED NOT HIDDEN.  Protection reuses the ring's epoch counter,
// so with RESC_GEN_RING <= 0 no line is ever stamped and NOTHING is ever
// protected.  Retention is then a no-op by construction.
// g_resc_retain_no_epoch counts that case so it is computable from any run
// rather than asserted.
//
// Knobs (unset == OFF == legacy, bit-identical):
//   RESC_RETAIN    0 = off (default), 1 = bias on
//   RESC_RETAIN_Q  quota, percent of ways protected lines may bias (default 75)
//   RESC_RETAIN_W  protection window in epochs (default = ring depth G)
// Counters are pure increments; they cannot feed back into any timing decision.
// ===========================================================================
static int g_resc_retain      = -1;   // -1 = not yet read from environment
static int g_resc_retain_qpct = 75;
static int g_resc_retain_win  = 0;    // 0 == follow ring depth G
unsigned long long g_resc_retain_victim_walks   = 0;  // capacity evictions seen
unsigned long long g_resc_retain_prot_seen      = 0;  // protected candidates seen
unsigned long long g_resc_retain_biased         = 0;  // bias FIRED (prot spared)
unsigned long long g_resc_retain_spills         = 0;  // protected line EVICTED
unsigned long long g_resc_retain_quota_override = 0;  // spill caused by quota
unsigned long long g_resc_retain_setfull        = 0;  // spill: no unprot victim
unsigned long long g_resc_retain_no_epoch       = 0;  // ring OFF => never protect
unsigned long long g_resc_retain_legacy_agree   = 0;  // 2nd pass == legacy pick
unsigned long long g_resc_retain_legacy_differ  = 0;  // MUST be 0 in every run
unsigned long long g_resc_retain_stamp_unmarked = 0;  // MUST be 0 in every run

static inline bool resc_retain_enabled() {
  if (g_resc_retain < 0) {
    const char *e = std::getenv("RESC_RETAIN");
    g_resc_retain = e ? atoi(e) : 0;
    if (g_resc_retain > 0) {
      const char *q = std::getenv("RESC_RETAIN_Q");
      if (q) g_resc_retain_qpct = atoi(q);
      const char *w = std::getenv("RESC_RETAIN_W");
      if (w) g_resc_retain_win = atoi(w);
      printf("\n[RESC_RETAIN] enabled=%d quota_pct=%d window=%d "
             "(window 0 = follow RESC_GEN_RING)\n",
             g_resc_retain, g_resc_retain_qpct, g_resc_retain_win);
    }
  }
  return g_resc_retain > 0;
}

// L1 data cache only.  L2 is constructed with core_id = -1 (l2cache.cc:439);
// the texture / constant / instruction L1s carry a type_id other than NORMALS
// (stat-tool.h:271; shader.cc get_shader_normal_cache_id() -> NORMALS).
static inline bool resc_retain_is_l1d(int core_id, int type_id) {
  return core_id >= 0 && type_id == (int)NORMALS;
}

// Protection window in epochs.  Default: the generation ring's own depth, so
// "protected" and "forwardable" release together -- one epoch notion, not two.
static inline unsigned long long resc_retain_window() {
  if (g_resc_retain_win > 0) return (unsigned long long)g_resc_retain_win;
  return (unsigned long long)(g_resc_gen_ring > 0 ? g_resc_gen_ring : 1);
}

// Does this line carry a producing generation stamp inside its live window?
// STAMP-FIRST: resc_stamp_gen() is reached only under
// should_set_cluster_state_for_write() (the marked-address test), so a stamped
// line is necessarily marked.  Registry membership is therefore CONFIRMED on
// the positive path rather than scanned on every call, and any disagreement is
// counted in g_resc_retain_stamp_unmarked, which must be 0 in every run.
static bool resc_retain_line_protected(new_addr_type block_addr) {
  if (g_resc_gen_ring <= 0) { g_resc_retain_no_epoch++; return false; }
  const unsigned long long W = resc_retain_window();
  for (auto &mi : g_resc_line_gen) {
    auto li = mi.second.find(block_addr);
    if (li == mi.second.end()) continue;
    auto ci = g_resc_cur_gen.find(mi.first);
    const unsigned long long cur =
        (ci != g_resc_cur_gen.end()) ? ci->second : 0ULL;
    const unsigned long long g = li->second;
    if (cur >= g && (cur - g) < W) {
      // Confirm the stamped-implies-marked invariant (computed, not assumed).
      bool marked = false;
      for (auto &kv : g_cluster_coherent_addrs_by_cluster) {
        if (kv.second.find(block_addr) != kv.second.end()) { marked = true; break; }
      }
      if (!marked) { g_resc_retain_stamp_unmarked++; return false; }
      return true;                       // inside the live window AND marked
    }
  }
  return false;   // never stamped, or the epoch has closed -> self-released
}

// Quota: may the bias be honoured with n_prot protected lines in this set?
static inline bool resc_retain_quota_ok(unsigned n_prot, unsigned assoc) {
  if (assoc == 0) return false;
  unsigned q = (unsigned)(((unsigned long long)assoc *
                           (unsigned long long)g_resc_retain_qpct) / 100ULL);
  if (q > assoc - 1) q = assoc - 1;   // never let the bias cover every way
  return n_prot <= q;
}

'''

# ---------------------------------------------------------------- B (decision)
# NOTE: the victim LOOP itself is deliberately left byte-unchanged.
B_OLD = """  if (invalid_line != (unsigned)-1) {
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    idx = valid_line;
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable

  return MISS;"""

B_NEW = """  if (invalid_line != (unsigned)-1) {
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    idx = valid_line;
    // =====================================================================
    // RESC EPOCH-SCOPED RETENTION BIAS (2026-09-04).  Reached ONLY here: the
    // legacy walk has already decided it must evict a VALID line, i.e. this is
    // a capacity eviction.  Default OFF -> the whole block is skipped and idx
    // keeps the legacy value assigned just above, so the default path is
    // bit-unchanged.
    //
    // Second pass over the SAME set, replicating the legacy eligibility test
    // verbatim and splitting the eligible candidates into protected and
    // unprotected.  Semantics are identical to classifying inside the first
    // walk; done here so the per-access hot loop keeps its original cost.
    // Charges no simulated cycles.
    // =====================================================================
    if (resc_retain_enabled() && resc_retain_is_l1d(m_core_id, m_type_id)) {
      const float __dirty_pct =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      unsigned __prot_line = (unsigned)-1, __unprot_line = (unsigned)-1;
      // NOTE: legacy initialises valid_timestamp to (unsigned)-1, NOT
      // ULLONG_MAX (gpu-cache.cc:747).  The second pass MUST use the identical
      // initialiser or the legacy_agree self-check could disagree spuriously.
      unsigned long long __prot_ts  = (unsigned)-1;
      unsigned long long __unprot_ts = (unsigned)-1;
      unsigned __n_prot = 0;
      for (unsigned way = 0; way < m_config.m_assoc; way++) {
        const unsigned index = set_index * m_config.m_assoc + way;
        cache_block_t *line = m_lines[index];
        // --- legacy eligibility, verbatim from the walk above ---
        if (line->is_reserved_line()) continue;
        if (line->is_modified_line() && __dirty_pct < m_config.m_wr_percent)
          continue;
        if (line->is_invalid_line()) continue;   // handled by invalid_line
        // --- classification ---
        const bool __rp = resc_retain_line_protected(line->m_block_addr);
        const unsigned long long __age =
            (m_config.m_replacement_policy == LRU) ? line->get_last_access_time()
                                                   : line->get_alloc_time();
        if (__rp) {
          __n_prot++;
          if (!probe_mode) g_resc_retain_prot_seen++;
          if (__age < __prot_ts) { __prot_ts = __age; __prot_line = index; }
        } else {
          if (__age < __unprot_ts) { __unprot_ts = __age; __unprot_line = index; }
        }
      }
      // Self-check, COMPUTED not asserted: with no protected line the second
      // pass must reproduce the legacy victim exactly.  legacy_differ must be
      // 0 in every reported run.
      if (__n_prot == 0 && !probe_mode) {
        if (__unprot_line == valid_line) g_resc_retain_legacy_agree++;
        else                             g_resc_retain_legacy_differ++;
      }
      if (__prot_line != (unsigned)-1) {
        if (!probe_mode) g_resc_retain_victim_walks++;
        const bool __quota_ok =
            resc_retain_quota_ok(__n_prot, m_config.m_assoc);
        if (__unprot_line != (unsigned)-1 && __quota_ok) {
          idx = __unprot_line;                 // SPARE the protected lines
          if (!probe_mode) g_resc_retain_biased++;
        } else {
          idx = __prot_line;                   // SPILL: evict oldest protected
          if (!probe_mode) {
            g_resc_retain_spills++;
            if (!__quota_ok) g_resc_retain_quota_override++;
            else             g_resc_retain_setfull++;
          }
        }
      }
    }
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable

  return MISS;"""

# ---------------------------------------------------------------- C (stats)
C_OLD = """  printf("gpu_tot_sim_insn = %lld\\n", gpu_tot_sim_insn + gpu_sim_insn);"""

C_NEW = """  {
    // RESC EPOCH-SCOPED RETENTION BIAS (2026-09-04): pure-increment counters.
    // biased = capacity evictions where a protected line was spared.
    // spills = protected lines evicted anyway -- retention's OWN cost, i.e.
    //          the handoff that now comes from L2 instead of a peer L1.
    // legacy_differ / stamp_unmarked are self-checks and MUST both be 0.
    extern unsigned long long g_resc_retain_victim_walks,
        g_resc_retain_prot_seen, g_resc_retain_biased, g_resc_retain_spills,
        g_resc_retain_quota_override, g_resc_retain_setfull,
        g_resc_retain_no_epoch, g_resc_retain_legacy_agree,
        g_resc_retain_legacy_differ, g_resc_retain_stamp_unmarked;
    unsigned long long __tot = g_resc_retain_biased + g_resc_retain_spills;
    printf("[RESC_RETAIN_STATS] victim_walks=%llu prot_candidates_seen=%llu "
           "biased=%llu spills=%llu spill_rate=%.6f quota_override=%llu "
           "set_full=%llu no_epoch=%llu legacy_agree=%llu legacy_differ=%llu "
           "stamp_unmarked=%llu\\n",
           g_resc_retain_victim_walks, g_resc_retain_prot_seen,
           g_resc_retain_biased, g_resc_retain_spills,
           __tot ? (double)g_resc_retain_spills / (double)__tot : 0.0,
           g_resc_retain_quota_override, g_resc_retain_setfull,
           g_resc_retain_no_epoch, g_resc_retain_legacy_agree,
           g_resc_retain_legacy_differ, g_resc_retain_stamp_unmarked);
  }
  printf("gpu_tot_sim_insn = %lld\\n", gpu_tot_sim_insn + gpu_sim_insn);"""

def main():
    p = ROOT + "gpu-cache.cc"
    s = rd(p)
    if "RESC EPOCH-SCOPED RETENTION BIAS" in s:
        sys.exit("REFUSING: retention code already present in gpu-cache.cc")
    s = sub1(s, A_ANCHOR, A_NEW + A_ANCHOR, "A/helpers")
    s = sub1(s, B_OLD, B_NEW, "B/decision")
    wr(p, s)
    print("PATCHED gpu-cache.cc")

    p = ROOT + "gpu-sim.cc"
    s = rd(p)
    if "RESC_RETAIN_STATS" in s:
        sys.exit("REFUSING: retention stats already present in gpu-sim.cc")
    s = sub1(s, C_OLD, C_NEW, "C/stats")
    wr(p, s)
    print("PATCHED gpu-sim.cc")

main()
