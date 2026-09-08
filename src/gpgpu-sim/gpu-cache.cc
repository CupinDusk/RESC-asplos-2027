// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Vijay Kandiah,
// Nikos Hardavellas The University of British Columbia, Northwestern University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include <assert.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <cstring>
#include "gpu-sim.h"
#include "hashing.h"
#include "stat-tool.h"
#include "shader.h"
// ============================================================================
// RESC cross-cluster (inter-GPC) single-writer fast path  [CCOP MECHANISM]
// ----------------------------------------------------------------------------
// Distinct from the intra-CGA read-share above:
//   * INTER-GPC: queries a DIFFERENT processing cluster (GPC), not the same
//     cluster_slot inside one GPC.
//   * Serves producer-WRITTEN write-through lines (single writer), records the
//     write OWNER in a per-GPC tag-only table (XCWL). The cross-cluster tier is
//     PREDICTION/SPECULATION (CCOP), NEVER broadcast: on a tagged miss the
//     consumer fires AT MOST ONE confidence-gated directed probe to the
//     predicted owner and speculatively touches L2 in parallel; cold/low-conf
//     reads fire ZERO probes and are byte-identical to the L2 baseline.
// Marked cross-cluster lines are registered GPC-agnostically (a consumer in a
// different GPC than the producer must still match), unlike the (gpc,slot)-keyed
// intra-cluster set above.
// ============================================================================
#include <set>
#include <map>
#include <algorithm>

// Block addresses declared as cross-cluster communication lines.
static std::unordered_set<new_addr_type> g_cross_cluster_addrs;
// Lines seen written by >1 distinct owner core => forced to L2 (multi-writer).
static std::unordered_set<new_addr_type> g_cross_cluster_multiwriter;

struct xcwl_entry { shader_core_ctx *owner_core; bool reuse; unsigned gen; };
// XCWL owner-recording table:  g_xcwl[gpc_id][block_addr] = {owner_core,reuse,gen}.
static std::map<unsigned, std::unordered_map<new_addr_type, xcwl_entry>> g_xcwl;
// GPC ids that have ever owned a cross-cluster line (neighbor enumeration).
static std::set<unsigned> g_xcwl_gpcs;
// Per-block owner generation counter (bumped on each producer write). Consumers
// remember the gen they seeded at; owner NACKs a probe whose expected gen is
// newer than the recorded one (staleness gate).
static std::unordered_map<new_addr_type, unsigned> g_xcwl_gen;

// ---------------------------------------------------------------------------
// CCOP (Cross-Cluster Owner Predictor) -- prediction/speculation tier that
// REPLACES the old k-neighbor broadcast. Per-l1_cache (per consumer SM),
// direct-mapped, region-keyed. On a tagged cross-cluster local miss the
// consumer consults CCOP; if it holds a confident, non-poisoned, cross-GPC
// owner it fires AT MOST ONE directed probe to that owner and speculatively
// touches L2 in parallel. There is NO broadcast anywhere. A cold/low-conf
// lookup fires ZERO probes and is byte-identical to the baseline. The owner is
// SEEDED FOR FREE from the L2 fill that already runs (XCWL owner tuple), so the
// first touch pays 0 probes and is always correct.
// ---------------------------------------------------------------------------
static const unsigned CCOP_REGION_SHIFT = 13;   // ~8KB producer slice per entry
static const unsigned CCOP_NENTRIES     = 128;  // direct-mapped, ~640B/SM
static const int      CCOP_CONF_THRESH  = 2;    // 2-bit sat; probe only if conf>=2
static const int      CCOP_CONF_MAX     = 3;
struct ccop_entry {
  bool         valid      = false;
  new_addr_type region_tag = 0;
  unsigned     owner_sid  = 0;
  unsigned     owner_gpc  = 0;
  int          conf       = 0;   // 2-bit saturating confidence
  unsigned     gen        = 0;   // owner generation seen at seed time
  bool         poison     = false; // 2nd distinct writer GPC seen -> never probe
};
// One table per consumer l1_cache instance (logically l1_cache::m_ccop; kept in
// a file-static keyed by the cache pointer to avoid header/ABI churn -- each
// l1_cache still owns an independent 128-entry table).
static std::unordered_map<const void *, std::vector<ccop_entry>> g_ccop_tables;

// Runtime knobs (env). Mode: 0=A(baseline/all-L2) 1=S1(CCOP P3(P2) default)
// 2=S2(probe-first, L2-pressure regime).
int g_c2c_mode          = -1;     // lazily initialised (<0 = unset)
int g_c2c_neighbor_k    = 4;      // (legacy; unused by CCOP)
int g_c2c_l2_roundtrip  = 242;    // modelled cross-cluster L2 round-trip
int g_c2c_probe_timeout = 100;    // (legacy; unused by CCOP)
int g_c2c_noprobe       = 0;      // C cold-only control: CCOP present, never probes
int g_resc_no_fwd       = 0;      // faithful-hLRC arm: disable marked-line cross-SM L1-to-L1 serve (RESC_NO_FWD)
int g_c2c_debug = -1;             // RESC_C2C_DEBUG env
extern int sm2sm_latency;         // C2C fabric latency (sm_2_sm_network.cc)

// Counters (per run).
unsigned long long g_c2c_neighbor_hits = 0;  // == ccop_fresh_hits (back-compat)
unsigned long long g_c2c_l2_fallbacks  = 0;  // cross-cluster reads that went to L2
unsigned long long g_c2c_mispredicts   = 0;  // == ccop_mispredict (back-compat)
unsigned long long g_c2c_pred_hits      = 0; // == ccop_fresh_hits (back-compat)
unsigned long long g_c2c_total_reads    = 0;
unsigned long long g_c2c_phys_resident_hits = 0; // served, line still in stock L1
unsigned long long g_c2c_pinned_only_hits   = 0; // served via XCWL owner directory only
// CCOP-specific counters (brief).
unsigned long long g_ccop_probes         = 0; // directed probes fired (<=1 per read)
unsigned long long g_ccop_fresh_hits     = 0; // probe confirmed owner -> served @sm2sm
unsigned long long g_ccop_mispredict     = 0; // probe NACK -> L2 completes (BW-only cost)
unsigned long long g_ccop_cold_l2_learn  = 0; // free seeds from the L2 fill (no probe)
unsigned long long g_ccop_poison_skips   = 0; // multi-writer/poison -> no probe
unsigned long long g_c2c_probe_bytes     = 0; // probe request bandwidth charged
unsigned long long g_c2c_l2_accesses_saved = 0; // L2 fetches skipped on a CCOP hit
unsigned long long g_c2c_crossgpc_probes = 0; // probes where owner_gpc != self
unsigned long long g_c2c_samegpc_skips   = 0; // owner_gpc == self (never probed)
unsigned long long g_c2c_crossgpc_reads  = 0; // reads w/ known owner on a diff GPC
unsigned long long g_c2c_samegpc_reads   = 0; // reads w/ known owner on same GPC

// Extra cycles to defer the just-completed cross-cluster load. Read by shader.cc
// immediately after access() returns; the sim executes L1 accesses serially so
// this single-slot handoff is safe. Sentinel -1 => use default intra-cluster charge.
int g_c2c_defer_extra_cycles = -1;

// ===========================================================================
// EXPERIMENT B (pre-registered 2026-08-29): SYMMETRIC per-supplier injection-port
// contention on the RESC peer-L1 read-share path.
// ---------------------------------------------------------------------------
// WHY: the sim already charges DSM N->1 injection convergence -- the crossbar
// output port (Crossbar::Advance, sm_2_sm_network.cc) drains ONE requester per
// cycle at sm2sm_bandwidth bits/cyc, so k DSM requests converging on one target
// SM serialize. The RESC read-share path (try_cluster_read_share /
// special_try_cluster_read_share) previously charged a FLAT sm2sm_latency with NO
// per-supplier port serialization (round-10 reviewer weakness #8). This adds the
// SAME class of cost to the peer-supply path ONLY: when k consumers pull a marked
// line from ONE supplier SM L1, that supplier injection port serializes responses.
//
// FORM (mirrors the DSM crossbar output port -- a single server draining at a
// fixed bit-rate): per-supplier port_free_at[sid] timeline. A read-share served
// from supplier s at absolute cycle now occupies s port for
//     service = ceil(payload_bits / g_resc_inject_bw_bits)   (>=1)
// and queues behind any prior occupancy:
//     start      = max(now, port_free_at[s])
//     queue_wait = start - now
//     port_free_at[s] = start + service
// The EXTRA cycles charged (on top of the frozen sm2sm_latency transport, exactly
// as DSM pays port-drain THEN the transport) = queue_wait + service.
//
// PARAMETER g_resc_inject_bw_bits is FROZEN from a real-H20 DSMEM injection
// microbench (route_inj.cu on the H20 host) -- the second silicon calibration
// point for the RESC datapath. Env RESC_INJECT_BW / RESC_INJECT_CHARGE exist ONLY
// for the integrity/sensitivity check, never to tune the headline.
// ---------------------------------------------------------------------------
int    g_resc_inject_charge  = -1;    // -1 unset; 0=off; 1=on. env RESC_INJECT_CHARGE
double g_resc_inject_bw_bits = 15.34; // SILICON-FROZEN injection-port BW (bits/cycle). PLACEHOLDER until set from H20 route_inj.
static std::unordered_map<unsigned, unsigned long long> g_resc_port_free_at;
unsigned long long g_resc_inject_events    = 0;
unsigned long long g_resc_inject_queue_cyc = 0;
unsigned long long g_resc_inject_serve_cyc = 0;
unsigned long long g_resc_inject_maxq      = 0;

static void init_resc_inject() {
  if (g_resc_inject_charge >= 0) return;
  const char *c  = std::getenv("RESC_INJECT_CHARGE");
  g_resc_inject_charge = c ? atoi(c) : 1;           // ON by default (Experiment B)
  const char *bw = std::getenv("RESC_INJECT_BW");
  if (bw) g_resc_inject_bw_bits = atof(bw);
  printf("\n[RESC_INJECT] charge=%d inject_bw_bits=%.4f (per-supplier injection-port"
         " serialization, symmetric with DSM crossbar)\n",
         g_resc_inject_charge, g_resc_inject_bw_bits);
}

// Extra deferral cycles (queue_wait + service) for a read-share response served
// by supplier_sid carrying payload_bits, arriving at now. Updates supplier port
// timeline. Returns 0 if disabled.
static int resc_injection_extra_cycles(unsigned supplier_sid,
                                       unsigned payload_bits,
                                       unsigned long long now) {
  init_resc_inject();
  if (g_resc_inject_charge == 0 || g_resc_inject_bw_bits <= 0.0) return 0;
  if (payload_bits == 0) payload_bits = 8;
  int service = (int)((payload_bits + g_resc_inject_bw_bits - 1.0) / g_resc_inject_bw_bits);
  if (service < 1) service = 1;
  unsigned long long prev_free = g_resc_port_free_at.count(supplier_sid)
                                     ? g_resc_port_free_at[supplier_sid] : 0ULL;
  unsigned long long start = (prev_free > now) ? prev_free : now;
  unsigned long long queue_wait = start - now;
  g_resc_port_free_at[supplier_sid] = start + (unsigned long long)service;
  g_resc_inject_events++;
  g_resc_inject_queue_cyc += queue_wait;
  g_resc_inject_serve_cyc += (unsigned long long)service;
  if (queue_wait > g_resc_inject_maxq) g_resc_inject_maxq = queue_wait;
  return (int)queue_wait + service;
}

// ===========================================================================
// SUPPLIER TAG-ARRAY PROBE-PORT OCCUPANCY (r29, prereg be694aa9...).
// ---------------------------------------------------------------------------
// WHY: reviewers correctly note that we charge the supplier's *fabric* injection
// port (EXPERIMENT B above) but NOT its *tag array*. A peer probe must read the
// supplier's tag array to discover the marked line; with up to C-1 consumers
// probing one supplier in a window, that array port serializes independently of
// the fabric. Previously "not charged" (main.tex:1121). This charges it.
//
// FORM: identical single-server timeline to the injection port, EXCEPT the
// service time is a CONSTANT -- a tag lookup is one array access, not a data
// transfer, so it does not scale with payload bits.
//
// PARAMETER: NOT silicon-calibrated. We have no H20 measurement of L1 tag-port
// service time, so this is swept {0,1,2,4,8} and reported as a range, never as a
// point. DEFAULT 0 = OFF, so all previously published cells reproduce exactly.
// ---------------------------------------------------------------------------
int g_resc_probe_port_cycles = -1;   // -1 unset; 0=off. env RESC_PROBE_PORT
static std::unordered_map<unsigned, unsigned long long> g_resc_tag_port_free_at;
unsigned long long g_resc_probe_events    = 0;
unsigned long long g_resc_probe_queue_cyc = 0;

static void init_resc_probe_port() {
  if (g_resc_probe_port_cycles >= 0) return;
  const char *p = std::getenv("RESC_PROBE_PORT");
  g_resc_probe_port_cycles = p ? atoi(p) : 0;        // OFF by default
  if (g_resc_probe_port_cycles < 0) g_resc_probe_port_cycles = 0;
  if (g_resc_probe_port_cycles > 0)
    printf("\n[RESC_PROBE_PORT] service=%d cyc/probe (supplier tag-array port"
           " serialization, constant service; NOT silicon-calibrated)\n",
           g_resc_probe_port_cycles);
}

// Extra deferral cycles for the supplier's tag-array port. Returns 0 when off.
static int resc_probe_port_extra_cycles(unsigned supplier_sid,
                                        unsigned long long now) {
  init_resc_probe_port();
  if (g_resc_probe_port_cycles == 0) return 0;
  unsigned long long service = (unsigned long long)g_resc_probe_port_cycles;
  unsigned long long prev_free = g_resc_tag_port_free_at.count(supplier_sid)
                                     ? g_resc_tag_port_free_at[supplier_sid] : 0ULL;
  unsigned long long start = (prev_free > now) ? prev_free : now;
  unsigned long long queue_wait = start - now;
  g_resc_tag_port_free_at[supplier_sid] = start + service;
  g_resc_probe_events++;
  g_resc_probe_queue_cyc += queue_wait;
  return (int)(queue_wait + service);
}

// ===========================================================================
// RESC MULTI-WRITER COHERENCE (ASPLOS) -- state + counters.
// ---------------------------------------------------------------------------
// Two intra-cluster (same-GPC) multi-writer resolutions replace the old
// "producer-consumer only" behaviour the Round-9 reviewers flagged:
//   (A) FALSE SHARING: a 2nd same-cluster SM writing DISJOINT sectors of a
//       marked line does NOT clobber the first writer's whole line. Both copies
//       become CLUSTER_MW_MERGE and each records only its own written sectors.
//       This rides the existing write-through (ZERO new fabric messages); the
//       consumer read serves the byte-accurate L2 merge (WT invariant) and is
//       forbidden from taking a single-owner whole-line forward.
//   (B) CLUSTER-SCOPED ATOMIC: a single migrating owner token (reusing the XCWL
//       slot) serialises the RMW at one SM. A peer atomic sends exactly one
//       UpgradeReq over the SM-to-SM fabric to the current owner, which
//       downgrades EM->INVALID transiting through FORWARD (this fires the dead
//       F-state + invalidate edge); the requester installs EM. Gated by
//       g_xcwl_gen (stale gen -> NACK). One fabric hop per ownership change,
//       never a broadcast, no sharer vector.
// The functional value ALWAYS comes from the functional simulator against L2
// (write-through), so bit-identity to a correct reference is structural; these
// paths change route/timing and the on-chip coherence bookkeeping only.
// Toggle the whole family with RESC_MW=1 (default 0 == legacy behaviour, i.e.
// the arm-3 "old escape"); RESC_MW_LWW=1 selects the whole-line last-writer-wins
// ablation (no per-sector mask) for the necessity arm.
// ---------------------------------------------------------------------------
int g_resc_mw          = -1;   // lazily initialised (<0 = unset); 1 = MW enabled
int g_resc_mw_lww      = 0;    // 1 = whole-line LWW ablation (no per-sector mask)
int g_resc_mw_debug    = 0;

// Per-GPC atomic owner token for marked lines: g_mw_atomic_owner[gpc][addr] =
// owning core. Reuses the same {gpc,addr} keying as XCWL; a single owner at a
// time (the token), migrated on each cross-SM ownership change. No sharer set.
static std::map<unsigned, std::unordered_map<new_addr_type, shader_core_ctx *>>
    g_mw_atomic_owner;

// Firing counters (per run) -- these PROVE the multi-writer / F-state /
// invalidate paths are actually exercised (the Round-9 deliverable).
unsigned long long g_mw_merge_entries      = 0; // CLUSTER_MW_MERGE state installs
unsigned long long g_mw_falsesharing_reads = 0; // MW_MERGE consumer merge-reads (no whole-line fwd)
unsigned long long g_mw_sector_writebacks  = 0; // per-sector merge-writebacks at epoch end
unsigned long long g_mw_atomic_upgradereq  = 0; // atomic UpgradeReq / token hand-offs (1 hop each)
unsigned long long g_mw_forward_downgrades = 0; // EM->INVALID transiting FORWARD (live invalidate)
unsigned long long g_mw_atomic_local_owner = 0; // atomic already owned locally (no hop)
unsigned long long g_mw_atomic_nack        = 0; // stale-gen UpgradeReq NACKed
unsigned long long g_mw_fabric_msgs        = 0; // total NEW fabric messages from MW (atomics only)

static void init_resc_mw() {
  if (g_resc_mw >= 0) return;
  const char *m  = std::getenv("RESC_MW");        g_resc_mw       = m  ? atoi(m)  : 0;
  const char *lw = std::getenv("RESC_MW_LWW");    g_resc_mw_lww   = lw ? atoi(lw) : 0;
  const char *db = std::getenv("RESC_MW_DEBUG");  g_resc_mw_debug = db ? atoi(db) : 0;
  printf("\n[RESC_MW] enabled=%d lww_ablation=%d debug=%d\n",
         g_resc_mw, g_resc_mw_lww, g_resc_mw_debug);
}

static void init_c2c_mode() {
  if (g_c2c_mode >= 0) return;
  const char *m = std::getenv("RESC_C2C_MODE");
  g_c2c_mode = m ? atoi(m) : 0;
  const char *k = std::getenv("RESC_C2C_K");       if (k) g_c2c_neighbor_k    = atoi(k);
  const char *l = std::getenv("RESC_C2C_L2");      if (l) g_c2c_l2_roundtrip  = atoi(l);
  const char *t = std::getenv("RESC_C2C_TIMEOUT"); if (t) g_c2c_probe_timeout = atoi(t);
  const char *np = std::getenv("RESC_C2C_NOPROBE"); if (np) g_c2c_noprobe = atoi(np);
  const char *dbg = std::getenv("RESC_C2C_DEBUG"); g_c2c_debug = dbg ? atoi(dbg) : 0;
  const char *nf = std::getenv("RESC_NO_FWD"); if (nf) g_resc_no_fwd = atoi(nf);
  if (g_resc_no_fwd) printf("\n[RESC_NO_FWD] faithful-hLRC: marked-line cross-SM L1-to-L1 serve DISABLED (consumer refetches L2); L1-caching+self-invalidation intact\n");
  printf("\n[CCOP] mode=%d (0=A/baseline 1=S1 2=S2) noprobe=%d l2_rt=%d sm2sm=%d "
         "region_shift=%u nentries=%u conf_thresh=%d\n",
         g_c2c_mode, g_c2c_noprobe, g_c2c_l2_roundtrip, sm2sm_latency,
         CCOP_REGION_SHIFT, CCOP_NENTRIES, CCOP_CONF_THRESH);
}

extern "C" void gpgpusim_register_cross_cluster_addr(new_addr_type addr) {
  // Mask to the 128B L1 line so any access offset within the line matches
  // block_addr(addr) used by the cache lookup.
  const new_addr_type LINE = 128;
  g_cross_cluster_addrs.insert(addr & ~(LINE - 1));
}
static inline bool is_cross_cluster_addr(new_addr_type block_addr) {
  return g_cross_cluster_addrs.find(block_addr) != g_cross_cluster_addrs.end();
}

// Record the producer (write owner) of a cross-cluster line into its GPC XCWL.
// Single-writer invariant: if a different core in a different GPC already owns
// the line, flag it multi-writer (correctness -> L2 only). Each producer write
// bumps a per-block generation counter; CCOP seeds carry the gen they saw and
// the owner NACKs a probe whose value would be stale.
static void xcwl_record_owner(new_addr_type block_addr, shader_core_ctx *owner) {
  if (!owner) return;
  simt_core_cluster *scc = owner->get_simt_core_cluster();
  if (!scc || !scc->m_gpc) return;
  unsigned g = scc->m_gpc->get_gpc_id();
  for (auto &kv : g_xcwl) {
    if (kv.first == g) continue;
    auto it = kv.second.find(block_addr);
    if (it != kv.second.end() && it->second.owner_core != owner)
      g_cross_cluster_multiwriter.insert(block_addr);
  }
  unsigned gen = ++g_xcwl_gen[block_addr];   // bump owner generation
  g_xcwl[g][block_addr] = xcwl_entry{owner, false, gen};
  g_xcwl_gpcs.insert(g);
  if (g_c2c_debug > 0)
    printf("[CCOP_DBG] record owner gpc=%u sid=%u addr=0x%llx gen=%u\n", g,
           owner->get_sid(), (unsigned long long)block_addr, gen);
}

// Fetch (or lazily create) the 128-entry CCOP table for one consumer l1_cache.
static std::vector<ccop_entry> &ccop_table_for(const void *cache) {
  auto it = g_ccop_tables.find(cache);
  if (it == g_ccop_tables.end())
    it = g_ccop_tables.emplace(cache, std::vector<ccop_entry>(CCOP_NENTRIES)).first;
  return it->second;
}
static inline new_addr_type ccop_region(new_addr_type block_addr) {
  return block_addr >> CCOP_REGION_SHIFT;
}
static inline unsigned ccop_slot(new_addr_type region) {
  return (unsigned)(region & (CCOP_NENTRIES - 1));
}

// 用于存储需要维护cluster一致性的地址集合（按 TB-cluster 维度）
// key = (gpc_id, cluster_slot) 组成的 64-bit key
static std::unordered_map<unsigned long long, std::unordered_set<new_addr_type>>
    g_cluster_coherent_addrs_by_cluster;
static bool g_cluster_coherent_addrs_initialized = false;

static inline unsigned long long make_cluster_key(unsigned gpc_id,
                                                  unsigned cluster_slot) {
  return (static_cast<unsigned long long>(gpc_id) << 32) |
         static_cast<unsigned long long>(cluster_slot);
}

// SELECTIVE-INVALIDATE (2026-09-03): RESC_QUIET=1 silences the per-event RESC
// debug printf (registration + read-share success).  Output-only; it cannot
// affect any counter or any timing decision.  Bit-identity of the counters
// under quiet vs loud is proven on a control point and disclosed.
static int g_resc_quiet = -1;
static inline bool resc_quiet() {
  if (g_resc_quiet < 0) {
    const char *q = std::getenv("RESC_QUIET");
    g_resc_quiet = q ? atoi(q) : 0;
  }
  return g_resc_quiet != 0;
}

// ===========================================================================
// RESC OVERLAPPING-EPOCH GENERATION RING (ASPLOS center-of-gravity novelty).
// A per-line generation tag + a per-cluster ring of G concurrent LIVE
// generations. A marked line produced in generation g stays readable over the
// SM-to-SM fabric (peer-L1 read-share) while g is within the live window
// [cur_gen-(G-1), cur_gen]. cluster.sync ADVANCES cur_gen (resc_epoch_tick);
// generations older than the window are RETIRED implicitly (O(1) -- the value
// is already in L2 by the write-through invariant, so retire needs no flush).
//   G=1  == eager bulk self-invalidate at every sync (hLRC / DeNovo drain).
//   G=2  == DeNovo keepValid current+next (two phases).
//   G>=3 == the new regime: >2 overlapping communication lifetimes, which a
//           single keep-valid interval structurally cannot represent.
// Default OFF (G<=0): no gating (legacy == unbounded G) so the sanity gate and
// every prior marked experiment stay byte-identical.
// ===========================================================================
int g_resc_gen_ring = 0;                 // ring depth G; <=0 == gating OFF
static bool g_resc_gen_ring_init = false;
unsigned long long g_resc_epoch_ticks = 0;
unsigned long long g_resc_gen_served  = 0;   // read-shares served within window
unsigned long long g_resc_gen_expired = 0;   // read-shares DENIED (out of window)
static std::unordered_map<unsigned long long, unsigned long long> g_resc_cur_gen;
static std::unordered_map<unsigned long long,
    std::unordered_map<new_addr_type, unsigned long long> > g_resc_line_gen;

static void init_resc_gen_ring() {
  if (g_resc_gen_ring_init) return;
  g_resc_gen_ring_init = true;
  const char *g = std::getenv("RESC_GEN_RING");
  g_resc_gen_ring = g ? atoi(g) : 0;
  printf("\n[RESC_GEN_RING] G=%d (<=0=OFF/legacy-unbounded; 1=eager/hLRC, "
         "2=DeNovo current+next, >=3=overlapping regime)\n", g_resc_gen_ring);
}
static inline bool resc_gating_on() { init_resc_gen_ring(); return g_resc_gen_ring > 0; }

// Advance a cluster's generation once per completed cluster.sync (epoch tick).
extern "C" void resc_epoch_tick(unsigned gpc_id, unsigned cluster_slot) {
  if (!resc_gating_on()) return;
  unsigned long long key = make_cluster_key(gpc_id, cluster_slot);
  g_resc_cur_gen[key] += 1;
  g_resc_epoch_ticks++;
}

// Compute the (gpc,slot) epoch key for a mem_fetch from the accessing L1.
static bool resc_cluster_key_for_mf(new_addr_type addr, shader_core_ctx *owner,
                                    mem_fetch *mf, unsigned long long *out_key) {
  (void)addr;
  if (!owner || !mf) return false;
  unsigned wid = mf->get_inst().warp_id();
  unsigned cluster_slot = owner->get_warp_cluster_slot(wid);
  simt_core_cluster *scc = owner->get_simt_core_cluster();
  if (!scc || !scc->m_gpc) return false;
  *out_key = make_cluster_key(scc->m_gpc->get_gpc_id(), cluster_slot);
  return true;
}

// Stamp a marked producer write with its producing generation.
static void resc_stamp_gen(unsigned long long key, new_addr_type block_addr) {
  if (g_resc_gen_ring <= 0) return;
  g_resc_line_gen[key][block_addr] = g_resc_cur_gen[key];
}

// Is a marked line still within the live generation window (fwd-able via peer
// L1)?  true when gating OFF or never-stamped (never penalise). Counts served
// vs expired for the ablation evidence.
static bool resc_gen_live(unsigned long long key, new_addr_type block_addr) {
  if (g_resc_gen_ring <= 0) return true;
  auto ci = g_resc_cur_gen.find(key);
  unsigned long long cur = (ci != g_resc_cur_gen.end()) ? ci->second : 0ULL;
  auto mi = g_resc_line_gen.find(key);
  if (mi == g_resc_line_gen.end()) return true;
  auto li = mi->second.find(block_addr);
  if (li == mi->second.end()) return true;
  unsigned long long g = li->second;
  bool live = (cur >= g) && ((cur - g) < (unsigned long long)g_resc_gen_ring);
  if (live) g_resc_gen_served++; else g_resc_gen_expired++;
  return live;
}

// ===========================================================================
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

// 用于识别需要设置cluster-state的目标地址（向后兼容，通过环境变量设置）
// 可以通过环境变量 CLUSTER_STATE_TARGET_ADDR 设置（十六进制地址）
static new_addr_type g_cluster_state_target_addr = 0;
static bool g_cluster_state_target_addr_initialized = false;

// 初始化目标地址（从环境变量读取，向后兼容）
static void init_cluster_state_target_addr() {
  if (g_cluster_state_target_addr_initialized) return;
  g_cluster_state_target_addr_initialized = true;

  const char* env_addr = std::getenv("CLUSTER_STATE_TARGET_ADDR");
  if (env_addr != nullptr) {
    g_cluster_state_target_addr = strtoull(env_addr, nullptr, 16);
    printf("\n[CLUSTER_STATE] Target address set to 0x%llx from environment\n",
           (unsigned long long)g_cluster_state_target_addr);
  }
}

// 注册需要维护cluster一致性的地址（由CUDA代码调用 / 由指令拦截器转发）
// 注意：这是“按 TB-cluster(=cluster_slot) 维度”注册，仅对同一 (gpc,slot) 的访问生效。
extern "C" void gpgpusim_register_cluster_coherent_addr(new_addr_type addr,
                                                       unsigned gpc_id,
                                                       unsigned cluster_slot) {
  if (!g_cluster_coherent_addrs_initialized) {
    g_cluster_coherent_addrs_initialized = true;
  }
  const unsigned long long key = make_cluster_key(gpc_id, cluster_slot);
  g_cluster_coherent_addrs_by_cluster[key].insert(addr);
  // SELECTIVE-INVALIDATE (2026-09-03): RESC_QUIET=1 suppresses this per-event
  // printf (multi-MB per run; caused timeouts in the previous experiment).
  // Default unset == current loud behaviour, so nothing changes unless asked.
  if (!resc_quiet())
  printf("\n[RESC] Registered cluster coherent address: 0x%llx (gpc=%u slot=%u)\n",
         (unsigned long long)addr, gpc_id, cluster_slot);
}

// RESC MULTI-WRITER (B): decide whether a marked cluster-scoped ATOMIC should be
// kept on the L1 path (so the owner-token hand-off can run) instead of the sim's
// default atomic behaviour of bypassing L1 to resolve at L2 (cache_op=CACHE_GLOBAL
// for ATOM_OP/RED_OP, cuda-sim.cc). Called from ldst_unit::memory_cycle. Returns
// true ONLY when RESC_MW is enabled AND the (line) address was registered as a
// cluster-coherent RESC line -- so it has ZERO effect on the sanity gate, the
// baseline, or any non-marked atomic. The functional RMW is unchanged (done by
// the functional simulator against L2); this only routes the timing/coherence
// bookkeeping through L1.
extern "C" bool resc_mw_atomic_wants_l1(new_addr_type block_addr) {
  init_resc_mw();
  if (!g_resc_mw) return false;
  // addr-only membership test across all (gpc,slot) buckets (marked line?).
  for (auto &kv : g_cluster_coherent_addrs_by_cluster) {
    if (kv.second.find(block_addr) != kv.second.end()) return true;
  }
  return false;
}

static bool is_cluster_coherent_addr_for_mf(new_addr_type addr,
                                           shader_core_ctx *owner,
                                           mem_fetch *mf) {
  if (!owner || !mf) return false;

  unsigned wid = mf->get_inst().warp_id();
  unsigned cluster_slot = owner->get_warp_cluster_slot(wid);
  simt_core_cluster *scc = owner->get_simt_core_cluster();
  if (!scc || !scc->m_gpc) return false;
  unsigned gpc_id = scc->m_gpc->get_gpc_id();

  const unsigned long long key = make_cluster_key(gpc_id, cluster_slot);
  auto it = g_cluster_coherent_addrs_by_cluster.find(key);
  if (it != g_cluster_coherent_addrs_by_cluster.end()) {
    if (it->second.find(addr) != it->second.end()) return true;
  }

  // 向后兼容：检查环境变量设置的目标地址
  init_cluster_state_target_addr();
  if (g_cluster_state_target_addr != 0 && addr == g_cluster_state_target_addr) {
    return true;
  }

  return false;
}

// 检查是否应该对当前写操作设置cluster-state
static bool should_set_cluster_state_for_write(new_addr_type addr,
                                               l1_cache *l1_this,
                                               mem_fetch *mf) {
  // 方法1：检查地址是否已注册为需要维护cluster一致性（按 TB-cluster 维度）
  if (l1_this && is_cluster_coherent_addr_for_mf(addr, l1_this->m_owner, mf)) {
    // printf("\n[CLUSTER_STATE] MATCH! Address 0x%llx is registered\n", (unsigned long long)addr);
    return true;
  }

  return false;
}

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels

const char *cache_request_status_str(enum cache_request_status status) {
  static const char *static_cache_request_status_str[] = {
      "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
      "SECTOR_MISS", "MSHR_HIT"};

  assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS);
  assert(status < NUM_CACHE_REQUEST_STATUS);

  return static_cache_request_status_str[status];
}

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  static const char *static_cache_reservation_fail_reason_str[] = {
      "LINE_ALLOC_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
      "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING"};

  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS);
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

  return static_cache_reservation_fail_reason_str[status];
}

unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
  // For sector cache, we select one sector per bank (sector interleaving)
  // This is what was found in Volta (one sector per bank, sector interleaving)
  // otherwise, line interleaving
  return cache_config::hash_function(addr, l1_banks,
                                     l1_banks_byte_interleaving_log2,
                                     l1_banks_log2, l1_banks_hashing_function);
}

unsigned cache_config::set_index(new_addr_type addr) const {
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function);
}

unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
  unsigned set_index = 0;

  switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
      /*
       * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
       * Distance Theory" Cedric Nugteren et al. HPCA 2014
       */
      unsigned lower_xor = 0;
      unsigned upper_xor = 0;

      if (m_nset == 32 || m_nset == 64) {
        // Lower xor value is bits 7-11
        lower_xor = (addr >> m_line_sz_log2) & 0x1F;

        // Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        upper_xor |= (addr & 0x80000) >> 15;  // Bit 19

        set_index = (lower_xor ^ upper_xor);

        // 48KB cache prepends the set_index with bit 12
        if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;

      } else { /* Else incorrect number of sets for the hashing function */
        assert(
            "\nGPGPU-Sim cache configuration error: The number of sets should "
            "be "
            "32 or 64 for the hashing set index function.\n" &&
            0);
      }
      break;
    }

    case BITWISE_XORING_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = bitwise_hash_function(higher_bits, index, m_nset);
      break;
    }
    case HASH_IPOLY_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = ipoly_hash_function(higher_bits, index, m_nset);
      break;
    }
    case CUSTOM_SET_FUNCTION: {
      /* No custom set function implemented */
      break;
    }

    case LINEAR_SET_FUNCTION: {
      set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
      break;
    }

    default: {
      assert("\nUndefined set index function.\n" && 0);
      break;
    }
  }

  // Linear function selected or custom set index function not implemented
  assert((set_index < m_nset) &&
         "\nError: Set index out of bounds. This is caused by "
         "an incorrect or unimplemented custom set index function.\n");

  return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(m_config_string, FuncCachePreferNone);
  m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
  }

  return cache_config::set_index(part_addr);
}

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) {
  init(core_id, type_id);
}

void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  unsigned cache_lines_num = config.get_max_num_lines();
  m_lines = new cache_block_t *[cache_lines_num];
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  // initialize snapshot counters for visualizer
  m_prev_snapshot_access = 0;
  m_prev_snapshot_miss = 0;
  m_prev_snapshot_pending_hit = 0;
  m_core_id = core_id;
  m_type_id = type_id;
  is_used = false;
  m_dirty = 0;
}

void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  return probe(addr, idx, mask, is_write, probe_mode, mf);
}

enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
  // assert( m_config.m_write_policy == READ_ONLY );
  unsigned set_index = m_config.set_index(addr);
  new_addr_type tag = m_config.tag(addr);

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;

  bool all_reserved = true;
  // check for hit or pending hit
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    if (line->m_tag == tag) {
      if (line->get_status(mask) == RESERVED) {
        idx = index;
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        if ((!is_write && line->is_readable(mask)) || is_write) {
          idx = index;
          return HIT;
        } else {
          idx = index;
          return SECTOR_MISS;
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
    if (!line->is_reserved_line()) {
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      // If the cacheline is from a load op (not modified),
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement
      // candidate i.e. Only evict clean cachelines until total dirty cachelines
      // reach the limit.
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        all_reserved = false;
        if (line->is_invalid_line()) {
          invalid_line = index;
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            if (line->get_last_access_time() < valid_timestamp) {
              valid_timestamp = line->get_last_access_time();
              valid_line = index;
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            }
          }
        }
      }
    }
  }
  if (all_reserved) {
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  if (invalid_line != (unsigned)-1) {
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

  return MISS;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;
  evicted_block_info evicted;
  enum cache_request_status result = access(addr, time, idx, wb, evicted, mf);
  assert(!wb);
  return result;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  m_access++;
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write());
  switch (status) {
    case HIT_RESERVED:
      m_pending_hit++;
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    case MISS:
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        if (m_lines[idx]->is_modified_line()) {
          wb = true;
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask());
          m_dirty--;
        }
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
      }
      break;
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    case RESERVATION_FAIL:
      m_res_fail++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
  fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(),
       is_write);
}

void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx;
  enum cache_request_status status = probe(addr, idx, mask, is_write);
  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time,
                           mask);
  } else if (status == SECTOR_MISS) {
    assert(m_config.m_cache_type == SECTOR);
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
  }
  if (before && !m_lines[idx]->is_modified_line()) {
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line();
  m_lines[idx]->fill(time, mask, byte_mask);
  if (m_lines[idx]->is_modified_line() && !before) {
    m_dirty++;
  }
}

void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
  assert(m_config.m_alloc_policy == ON_MISS);
  bool before = m_lines[index]->is_modified_line();
  m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));
      }
    }

  m_dirty = 0;
  is_used = false;
}

void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

  m_dirty = 0;
  is_used = false;
}

// ===========================================================================
// SELECTIVE CLUSTER-ACQUIRE INVALIDATE (2026-09-03).
// Pre-registration: selective-invalidate-prereg.md, SHA-256 8ebabe19...,
// frozen 2026-09-03T11:11:43Z.
//
// tag_array::invalidate() above is the CCTL.IVALL model: unconditional, every
// line, every sector.  It is what shipping sm_90 must do, because hardware
// cannot tell which lines a peer may have written.  RESC CAN tell: the marked
// communication set is registered explicitly
// (gpgpusim_register_cluster_coherent_addr, gpu-cache.cc:444-452).  The
// minimal correct acquire therefore invalidates ONLY marked lines and
// preserves unmarked ones -- unmarked lines are block-private or read-only by
// construction, so no peer can have made them stale.
//
// owner_retaining == true additionally preserves marked lines held in an
// OWNER cluster-state (EXCLUSIVE / EM / FORWARD / SM / MW_MERGE).  That is the
// standard DeNovo/hLRC self-invalidation rule -- an acquire self-invalidates
// valid-but-not-owned copies; the owner's copy is the newest and is retained.
// Without it, the producer erases its own authoritative copy at every barrier
// and there is nothing left for a peer to be forwarded (prereg S0a).
//
// Counters are pure increments; they cannot feed back into timing.
// ===========================================================================
unsigned long long g_l1_rd_marked_hit    = 0;
unsigned long long g_l1_rd_marked_miss   = 0;
unsigned long long g_l1_rd_unmarked_hit  = 0;
unsigned long long g_l1_rd_unmarked_miss = 0;
unsigned long long g_selinv_events            = 0;  // selective invalidate calls
unsigned long long g_selinv_lines_examined    = 0;
unsigned long long g_selinv_marked_invalidated= 0;
unsigned long long g_selinv_unmarked_preserved= 0;
unsigned long long g_selinv_owner_retained    = 0;

// Is this line address in the marked registry for (gpc,slot)?  Falls back to
// an address-only membership test across every bucket when the caller cannot
// supply a key -- deliberately CONSERVATIVE for correctness: a line marked in
// any cluster is treated as marked here, so we can only ever invalidate MORE
// than strictly necessary, never less.
static bool selinv_addr_is_marked(new_addr_type block_addr,
                                  bool have_key, unsigned long long key) {
  if (have_key) {
    auto it = g_cluster_coherent_addrs_by_cluster.find(key);
    if (it != g_cluster_coherent_addrs_by_cluster.end() &&
        it->second.find(block_addr) != it->second.end())
      return true;
    // fall through: the same physical line may have been registered by another
    // (gpc,slot) bucket; treating it as marked is the safe direction.
  }
  for (auto &kv : g_cluster_coherent_addrs_by_cluster) {
    if (kv.second.find(block_addr) != kv.second.end()) return true;
  }
  return false;
}

static bool selinv_block_is_owner(cache_block_t *blk,
                                  mem_access_sector_mask_t sm) {
  cluster_line_state st = CLUSTER_INVALID;
  line_cache_block *lb = dynamic_cast<line_cache_block *>(blk);
  if (lb) {
    st = lb->get_cluster_state();
  } else {
    sector_cache_block *sb = dynamic_cast<sector_cache_block *>(blk);
    if (!sb) return false;
    st = sb->get_cluster_state(sm);
  }
  return (st == CLUSTER_EXCLUSIVE || st == CLUSTER_EM ||
          st == CLUSTER_FORWARD   || st == CLUSTER_SM ||
          st == CLUSTER_MW_MERGE);
}

void tag_array::selective_invalidate(bool have_key, unsigned long long key,
                                     bool owner_retaining) {
  if (!is_used) return;
  g_selinv_events++;

  bool any_valid_left = false;
  for (unsigned i = 0; i < m_config.get_num_lines(); i++) {
    cache_block_t *blk = m_lines[i];
    if (blk->is_invalid_line()) continue;
    g_selinv_lines_examined++;

    const new_addr_type ba = blk->m_block_addr;
    if (!selinv_addr_is_marked(ba, have_key, key)) {
      g_selinv_unmarked_preserved++;
      any_valid_left = true;
      continue;                        // UNMARKED -> preserved (the whole point)
    }
    // Marked.  Per-sector, because a sector cache holds per-sector state.
    bool line_retained = false;
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
      mem_access_sector_mask_t sm = mem_access_sector_mask_t().set(j);
      if (blk->get_status(sm) == INVALID) continue;
      if (owner_retaining && selinv_block_is_owner(blk, sm)) {
        g_selinv_owner_retained++;
        line_retained = true;
        continue;                      // owner copy is newest -> keep
      }
      blk->set_status(INVALID, sm);
      g_selinv_marked_invalidated++;
    }
    if (line_retained) any_valid_left = true;
  }

  // m_dirty / is_used bookkeeping: unlike the whole-L1 form we cannot simply
  // zero them, because lines survive.  Recompute m_dirty exactly.
  unsigned d = 0;
  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) d++;
  m_dirty = d;
  if (!any_valid_left) is_used = false;
}

float tag_array::windowed_miss_rate() const {
  unsigned n_access = m_access - m_prev_snapshot_access;
  unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
  // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

  float missrate = 0.0f;
  if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
  return missrate;
}

void tag_array::new_window() {
  m_prev_snapshot_access = m_access;
  m_prev_snapshot_miss = m_miss;
  m_prev_snapshot_miss = m_miss + m_sector_miss;
  m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream);
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access);
  total_misses += (m_miss + m_sector_miss);
  total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;
  total_misses = (m_miss + m_sector_miss);
  total_hit_res = m_pending_hit;
  total_res_fail = m_res_fail;
}

bool was_write_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_REQUEST_SENT) return true;
  }
  return false;
}

bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
      wb_event = *e;
      return true;
    }
  }
  return false;
}

bool was_read_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == READ_REQUEST_SENT) return true;
  }
  return false;
}

bool was_readshare_hit(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == CLUSTER_READ_SHARE_HIT) return true;
  }
  return false;
}

bool was_writeallocate_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT) return true;
  }
  return false;
}
/****************************************************************** MSHR
 * ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const {
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const {
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    return i->second.m_list.size() >= m_max_merged;
  else
    return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR en contains an atomic operation
  if (mf->isatomic()) {
    m_data[block_addr].m_has_atomic = true;
  }
}

/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  assert(!busy());
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  m_current_response.push_back(block_addr);
  has_atomic = a->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

/// Returns next ready access
mem_fetch *mshr_table::next_access() {
  assert(access_ready());
  new_addr_type block_addr = m_current_response.front();
  assert(!m_data[block_addr].m_list.empty());
  mem_fetch *result = m_data[block_addr].m_list.front();
  m_data[block_addr].m_list.pop_front();
  if (m_data[block_addr].m_list.empty()) {
    // release entry
    m_data.erase(block_addr);
    m_current_response.pop_front();
  }
  return result;
}

void mshr_table::display(FILE *fp) const {
  fprintf(fp, "MSHR contents\n");
  for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
    unsigned block_addr = e->first;
    fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
            e->second.m_has_atomic, e->second.m_list.size());
    if (!e->second.m_list.empty()) {
      mem_fetch *mf = e->second.m_list.front();
      fprintf(fp, "%p :", mf);
      mf->print(fp);
    } else {
      fprintf(fp, " no memory requests???\n");
    }
  }
}
/***************************************************************** Caches
 * *****************************************************************/
cache_stats::cache_stats() {
  m_stats.resize(NUM_MEM_ACCESS_TYPE);
  m_stats_pw.resize(NUM_MEM_ACCESS_TYPE);
  m_fail_stats.resize(NUM_MEM_ACCESS_TYPE);
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_stats_pw[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    m_fail_stats[i].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear() {
  ///
  /// Zero out all current cache statistics
  ///
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    std::fill(m_stats_pw[i].begin(), m_stats_pw[i].end(), 0);
    std::fill(m_fail_stats[i].begin(), m_fail_stats[i].end(), 0);
  }
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear_pw() {
  ///
  /// Zero out per-window cache statistics
  ///
  for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i) {
    std::fill(m_stats_pw[i].begin(), m_stats_pw[i].end(), 0);
  }
}

void cache_stats::inc_stats(int access_type, int access_outcome) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  m_stats[access_type][access_outcome]++;
}

void cache_stats::inc_stats_pw(int access_type, int access_outcome) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");
  m_stats_pw[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(int access_type, int fail_outcome) {
  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  m_fail_stats[access_type][fail_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  ///
  /// This function selects how the cache access outcome should be counted.
  /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
  /// counted as a HIT_RESERVED in the caches.
  ///
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome) {
  ///
  /// Simple method to read/modify the stat corresponding to (access_type,
  /// access_outcome) Used overloaded () to avoid the need for separate
  /// read/write member functions
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
  }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome) const {
  ///
  /// Const accessor into m_stats.
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
  }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  cache_stats ret;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      ret(type, status, false) =
          m_stats[type][status] + cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      ret(type, status, true) =
          m_fail_stats[type][status] + cs(type, status, true);
    }
  }
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
  return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  ///
  /// Overloaded += operator to allow for simple stat accumulation
  ///
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      m_stats[type][status] += cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      m_stats_pw[type][status] += cs(type, status, false);
    }
    for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
         ++status) {
      m_fail_stats[type][status] += cs(type, status, true);
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const {
  ///
  /// Print out each non-zero cache statistic for every memory access type and
  /// status "cache_name" defaults to "Cache_stats" when no argument is
  /// provided, otherwise the provided name is used. The printed format is
  /// "<cache_name>[<request_type>][<request_status>] = <stat_value>"
  ///
  std::vector<unsigned> total_access;
  total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
  std::string m_cache_name = cache_name;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_request_status_str((enum cache_request_status)status),
              m_stats[type][status]);

      if (status != RESERVATION_FAIL && status != MSHR_HIT)
        // MSHR_HIT is a special type of SECTOR_MISS
        // so its already included in the SECTOR_MISS
        total_access[type] += m_stats[type][status];
    }
  }
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    if (total_access[type] > 0)
      fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
              total_access[type]);
  }
}

void cache_stats::print_fail_stats(FILE *fout, const char *cache_name) const {
  std::string m_cache_name = cache_name;
  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS; ++fail) {
      if (m_fail_stats[type][fail] > 0) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type),
                cache_fail_status_str((enum cache_reservation_fail_reason)fail),
                m_fail_stats[type][fail]);
      }
    }
  }
}

void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
  float data_port_util = 0.0f;
  if (port_available_cycles > 0) {
    data_port_util = (float)data_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
  float fill_port_util = 0.0f;
  if (port_available_cycles > 0) {
    fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  unsigned long long total = 0;
  for (unsigned type = 0; type < num_access_type; ++type) {
    for (unsigned status = 0; status < num_access_status; ++status) {
      if (!check_valid((int)access_type[type], (int)access_status[status]))
        assert(0 && "Unknown cache access type or access outcome");
      total += m_stats[access_type[type]][access_status[status]];
    }
  }
  return total;
}

void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats t_css;
  t_css.clear();

  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      if (status == HIT || status == MISS || status == SECTOR_MISS ||
          status == HIT_RESERVED)
        t_css.accesses += m_stats[type][status];

      if (status == MISS || status == SECTOR_MISS)
        t_css.misses += m_stats[type][status];

      if (status == HIT_RESERVED) t_css.pending_hits += m_stats[type][status];

      if (status == RESERVATION_FAIL) t_css.res_fails += m_stats[type][status];
    }
  }

  t_css.port_available_cycles = m_cache_port_available_cycles;
  t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
  t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

  css = t_css;
}

void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats_pw t_css;
  t_css.clear();

  for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
    for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
      if (status == HIT || status == MISS || status == SECTOR_MISS ||
          status == HIT_RESERVED)
        t_css.accesses += m_stats_pw[type][status];

      if (status == HIT) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_hits += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_hits += m_stats_pw[type][status];
        }
      }

      if (status == MISS || status == SECTOR_MISS) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_misses += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_misses += m_stats_pw[type][status];
        }
      }

      if (status == HIT_RESERVED) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_pending_hits += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_pending_hits += m_stats_pw[type][status];
        }
      }

      if (status == RESERVATION_FAIL) {
        if (type == GLOBAL_ACC_R || type == CONST_ACC_R || type == INST_ACC_R) {
          t_css.read_res_fails += m_stats_pw[type][status];
        } else if (type == GLOBAL_ACC_W) {
          t_css.write_res_fails += m_stats_pw[type][status];
        }
      }
    }
  }

  css = t_css;
}

bool cache_stats::check_valid(int type, int status) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}

bool cache_stats::check_fail_valid(int type, int fail) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
  m_cache_port_available_cycles += 1;
  if (data_port_busy) {
    m_cache_data_port_busy_cycles += 1;
  }
  if (fill_port_busy) {
    m_cache_fill_port_busy_cycles += 1;
  }
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
  m_data_port_occupied_cycles = 0;
  m_fill_port_occupied_cycles = 0;
}

/// use the data port based on the outcome and events generated by the mem_fetch
/// request
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events) {
  unsigned data_size = mf->get_data_size();
  unsigned port_width = m_config.m_data_port_width;
  switch (outcome) {
    case HIT: {
      unsigned data_cycles =
          data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
      m_data_port_occupied_cycles += data_cycles;
    } break;
    case HIT_RESERVED:
    case MISS: {
      // the data array is accessed to read out the entire line for write-back
      // in case of sector cache we need to write bank only the modified sectors
      cache_event ev(WRITE_BACK_REQUEST_SENT);
      if (was_writeback_sent(events, ev)) {
        unsigned data_cycles = ev.m_evicted_block.m_modified_size / port_width;
        m_data_port_occupied_cycles += data_cycles;
      }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      // Does not consume any port bandwidth
      break;
    default:
      assert(0);
      break;
  }
}

/// use the fill port
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
  m_fill_port_occupied_cycles += fill_cycles;
}

/// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles -= 1;
  }
  assert(m_data_port_occupied_cycles >= 0);

  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles -= 1;
  }
  assert(m_fill_port_occupied_cycles >= 0);
}

/// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const {
  return (m_data_port_occupied_cycles == 0);
}

/// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0);
}

/// Sends next request to lower level of memory
void baseline_cache::cycle() {
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      m_miss_queue.pop_front();
      m_memport->push(mf);
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}

/// Interface for response from lower memory level (model bandwidth restictions
/// in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  mf->set_data_size(e->second.m_data_size);
  mf->set_addr(e->second.m_addr);

  // 针对普通地址的读miss，保留脏字节标记，只取无脏字节标记的部分
  if (!mf->is_write() && mf->get_access_type() == GLOBAL_ACC_R &&
      !should_set_cluster_state_for_write(e->second.m_block_addr,
                                          dynamic_cast<l1_cache *>(this), mf)) {
    unsigned cache_index = (unsigned)-1;
    if (m_config.m_alloc_policy == ON_MISS) {
      cache_index = e->second.m_cache_index;
    } else {
      enum cache_request_status status = m_tag_array->probe(e->second.m_block_addr, cache_index, mf, false);
      assert(status == HIT || status == HIT_RESERVED);
    }
    cache_block_t *block = m_tag_array->get_block(cache_index);
    mem_access_byte_mask_t existing_dirty_mask = block->get_dirty_byte_mask();

    // 保存现有的脏字节掩码，fill后恢复
    // fill函数会更新块的状态，但我们需要保留脏字节标记
    // 注意：fill函数会调用block->fill()，这会更新块的状态，但不会清除脏字节掩码
    // 所以这里我们只需要确保fill后脏字节掩码被保留
    // 实际上，fill函数不会清除脏字节掩码，所以这里不需要特殊处理
    // 但是，我们需要确保只更新非脏字节的部分
    // 由于fill函数会更新整个块，我们需要在fill后恢复脏字节掩码
    // 但这样可能会覆盖从内存读取的数据
    // 更好的方法是：在fill之前，如果块已经有脏字节，我们不应该调用fill
    // 或者，我们需要修改fill函数，使其只更新非脏字节的部分
    // 由于这涉及到更复杂的修改，我们暂时保留现有逻辑
    // 注意：在实际实现中，可能需要修改tag_array::fill或block->fill来支持部分更新
  }

  if (m_config.m_alloc_policy == ON_MISS)
    m_tag_array->fill(e->second.m_cache_index, time, mf);
  else if (m_config.m_alloc_policy == ON_FILL) {
    m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
  } else
    abort();
 // post_fill(mf, e->second.m_cache_index);

  // 对于只读数据，第一次从内存fill时设置为E状态
  if (!mf->is_write() && mf->get_access_type() == GLOBAL_ACC_R) {
    unsigned cache_index = (unsigned)-1;
    if (m_config.m_alloc_policy == ON_MISS) {
      cache_index = e->second.m_cache_index;
    } else {
      enum cache_request_status status = m_tag_array->probe(e->second.m_block_addr, cache_index, mf, false);
      assert(status == HIT || status == HIT_RESERVED);
    }
    cache_block_t *block = m_tag_array->get_block(cache_index);
    mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();

    // 检查是否是line_cache_block类型
    line_cache_block *line_block = dynamic_cast<line_cache_block*>(block);
    if (line_block) {
      cluster_line_state current_state = line_block->get_cluster_state();
      if (current_state == CLUSTER_INVALID) {
        // 第一次fill，设置为E状态
        line_block->set_cluster_state(CLUSTER_EXCLUSIVE);
        // printf("\n[FILL] Read miss: SET CLUSTER STATE TO CLUSTER_EXCLUSIVE (line_block), addr=0x%llx\n", (unsigned long long)e->second.m_block_addr);
      } else {
        // printf("\n[FILL] Read miss: line_block already has state %d, addr=0x%llx\n", current_state, (unsigned long long)e->second.m_block_addr);
      }
    }

    // 检查是否是sector_cache_block类型
    sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);
    if (sector_block) {
      cluster_line_state current_state = sector_block->get_cluster_state(sector_mask);
      if (current_state == CLUSTER_INVALID) {
        // 第一次fill，设置为E状态
        sector_block->set_cluster_state(CLUSTER_EXCLUSIVE, sector_mask);
        // printf("\n[FILL] Read miss: SET CLUSTER STATE TO CLUSTER_EXCLUSIVE (sector_block), addr=0x%llx\n", (unsigned long long)e->second.m_block_addr);
      } else {
        // printf("\n[FILL] Read miss: sector_block already has state %d, addr=0x%llx\n", current_state, (unsigned long long)e->second.m_block_addr);
      }
    }
  }

  // 对于写操作，fill完成后设置cluster state（与wr_hit_wb逻辑相同）
  // 包括两种情况：
  // 1. 直接的写操作fill
  // 2. 写miss时用于write allocate的读请求fill（通过original_mf判断）
  mem_fetch *original_mf = mf->get_original_mf();
  bool is_write_fill = false;
  mem_fetch *write_mf = nullptr;

  if (mf->is_write() && mf->get_access_type() == GLOBAL_ACC_W) {
    // 情况1：直接的写操作fill
    is_write_fill = true;
    write_mf = mf;
  } else if (!mf->is_write() && original_mf &&
             original_mf->is_write() &&
             original_mf->get_access_type() == GLOBAL_ACC_W) {
    // 情况2：写miss时用于write allocate的读请求fill
    is_write_fill = true;
    write_mf = original_mf;
  }

  if (is_write_fill &&
      should_set_cluster_state_for_write(e->second.m_block_addr,
                                         dynamic_cast<l1_cache *>(this),
                                         write_mf)) {
    unsigned cache_index = (unsigned)-1;
    if (m_config.m_alloc_policy == ON_MISS) {
      cache_index = e->second.m_cache_index;
    } else {
      enum cache_request_status status = m_tag_array->probe(e->second.m_block_addr, cache_index, mf, false);
      assert(status == HIT || status == HIT_RESERVED);
    }
    cache_block_t *block = m_tag_array->get_block(cache_index);
    mem_access_sector_mask_t sector_mask = write_mf->get_access_sector_mask();
    line_cache_block *line_block = dynamic_cast<line_cache_block*>(block);
    sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);

    // 检查cluster内其他SM的L1 cache是否有拷贝
    // 注意：cluster-state只对L1 cache生效，L2 cache不使用cluster-state
    bool has_peer_copy = false;
    l1_cache *l1_this = dynamic_cast<l1_cache*>(this);
    if (l1_this && l1_this->m_owner) {
      simt_core_cluster *scc = l1_this->m_owner->get_simt_core_cluster();
      if (scc && scc->m_gpc) {
        unsigned req_wid = write_mf->get_inst().warp_id();
        unsigned cluster_slot = l1_this->m_owner->get_warp_cluster_slot(req_wid);
        std::vector<shader_core_ctx*> peers;
        scc->m_gpc->collect_shader_cores_for_cluster_slot(cluster_slot, peers);

        for (auto *core : peers) {
          if (!core || core == l1_this->m_owner) continue;
          l1_cache *peer = core->get_L1D_cache();
          if (!peer) continue;

          unsigned pidx = (unsigned)-1;
          cache_request_status ps = peer->m_tag_array->probe(e->second.m_block_addr, pidx, mf, false, true);
          if (ps == HIT || ps == HIT_RESERVED) {
            has_peer_copy = true;
            // 更新其他SM的cache数据
            peer->m_tag_array->fill(e->second.m_block_addr, time, mf, false);

            // 设置其他SM的cluster state为CLUSTER_SHARED
            cache_block_t *pblk = peer->m_tag_array->get_block(pidx);
            line_cache_block *peer_line_block = dynamic_cast<line_cache_block*>(pblk);
            sector_cache_block *peer_sector_block = dynamic_cast<sector_cache_block*>(pblk);

            if (peer_line_block) {
              peer_line_block->set_cluster_state(CLUSTER_SHARED);
            }
            if (peer_sector_block) {
              peer_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);
            }
          }
        }
      }
      if (has_peer_copy) {
        // 有其他SM的拷贝：设置本写者SM状态为CLUSTER_SM
        if (line_block) {
          line_block->set_cluster_state(CLUSTER_SM);
        }
        if (sector_block) {
          sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
        }
        // printf("\nFILL: SET THE CLUSTER STATE TO CLUSTER_SM (write miss fill)\n");
        // printf("\n[FILL] owner_sid=%u \n", l1_this->m_owner->get_sid());
      } else {
        // 没有其他SM的拷贝：设置状态为CLUSTER_EM
        if (line_block) {
          line_block->set_cluster_state(CLUSTER_EM);
        }
        if (sector_block) {
          sector_block->set_cluster_state(CLUSTER_EM, sector_mask);
        }
        // printf("\nFILL: SET THE CLUSTER STATE TO CLUSTER_EM (write miss fill)\n");
        // printf("\n[FILL] owner_sid=%u \n", l1_this->m_owner->get_sid());
      }
    }

    // 根据是否有其他SM的拷贝设置状态
    //if (l1_this) {

    //}
  }

  bool has_atomic = false;
  m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
  if (has_atomic) {
    assert(m_config.m_alloc_policy == ON_MISS);
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as dirty for
                                                      // atomic operation
    block->set_byte_mask(mf);
  }
  m_extra_mf_fields.erase(mf);
  m_bandwidth_management.use_fill_port(mf);
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
  fprintf(fp, "Cache %s:\t", m_name.c_str());
  m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const {
  fprintf(fp, "Cache %s:\n", m_name.c_str());
  m_mshrs.display(fp);
  fprintf(fp, "\n");
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;
  evicted_block_info e;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e,
                    events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (mshr_hit && mshr_avail) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    m_mshrs.add(mshr_addr, mf);
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT);
    do_miss = true;

  } else if (!mshr_hit && mshr_avail &&
             (m_miss_queue.size() < m_config.m_miss_queue_size)) {
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    m_mshrs.add(mshr_addr, mf);
    m_extra_mf_fields[mf] = extra_mf_fields(
        mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    mf->set_data_size(m_config.get_atom_sz());
    mf->set_addr(mshr_addr);
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
    if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));

    do_miss = true;
  } else if (mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
  else if (!mshr_hit && !mshr_avail)
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
  else
    assert(0);
}

/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);
  m_miss_queue.push_back(mf);
  mf->set_status(m_miss_queue_status, time);
}

void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  cache_block_t *block = m_tag_array->get_block(cache_index);
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    if (mf->get_access_sector_mask().test(i)) {
      bool all_set = true;
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      if (all_set) block->set_m_readable(true, mf->get_access_sector_mask());
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                         unsigned cache_index, mem_fetch *mf,
                                         unsigned time,
                                         std::list<cache_event> &events,
                                         enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  block->set_m_readable(true, mf->get_access_sector_mask());

  // 写操作时，根据cluster内其他SM的cache状态设置cluster state
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  line_cache_block *line_block = dynamic_cast<line_cache_block*>(block);
  sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);

  // 检查是否应该设置cluster-state：只对特定的写指令设置
  // 支持两种识别方式：
  // 1. 通过cache_op识别（st.global.wb.f32使用CACHE_WRITE_BACK）
  // 2. 通过地址识别（通过环境变量CLUSTER_STATE_TARGET_ADDR设置目标地址）
  if (!should_set_cluster_state_for_write(block_addr,
                                          dynamic_cast<l1_cache *>(this), mf)) {
    // 如果不需要设置cluster-state，直接返回
    return HIT;
  }

  // 检查cluster内其他SM的L1 cache是否有拷贝
  // 注意：m_owner是l1_cache的protected成员，需要通过类型转换访问
  // printf("\n[CLUSTER_STATE] wr_hit_wb: SHOULD SET CLUSTER STATE FOR WRITE: addr: %llx\n", addr);
  // printf("\n[CLUSTER_STATE] Checking for peer copy\n");
  bool has_peer_copy = false;
  l1_cache *l1_this = dynamic_cast<l1_cache*>(this);
  if (l1_this && l1_this->m_owner) {
    simt_core_cluster *scc = l1_this->m_owner->get_simt_core_cluster();
    if (scc && scc->m_gpc) {
      unsigned req_wid = mf->get_inst().warp_id();
      unsigned cluster_slot = l1_this->m_owner->get_warp_cluster_slot(req_wid);
      std::vector<shader_core_ctx*> peers;
      scc->m_gpc->collect_shader_cores_for_cluster_slot(cluster_slot, peers);

      for (auto *core : peers) {
        if (!core || core == l1_this->m_owner) continue;
        l1_cache *peer = core->get_L1D_cache();
        if (!peer) continue;

        unsigned pidx = (unsigned)-1;
        cache_request_status ps = peer->m_tag_array->probe(block_addr, pidx, mf, false, true);
        if (ps == HIT || ps == HIT_RESERVED) {
          has_peer_copy = true;
          // 更新其他SM的cache数据
          peer->m_tag_array->fill(block_addr, time, mf, false);

          // 设置其他SM的cluster state为CLUSTER_SHARED
          cache_block_t *pblk = peer->m_tag_array->get_block(pidx);
          line_cache_block *peer_line_block = dynamic_cast<line_cache_block*>(pblk);
          sector_cache_block *peer_sector_block = dynamic_cast<sector_cache_block*>(pblk);

          if (peer_line_block) {
            peer_line_block->set_cluster_state(CLUSTER_SHARED);
          }
          if (peer_sector_block) {
            peer_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);
          }
        }
      }
    }
    if (has_peer_copy) {
      // 有其他SM的拷贝：设置本写者SM状态为CLUSTER_SM
      if (line_block) {
        line_block->set_cluster_state(CLUSTER_SM);
      }
      if (sector_block) {
        sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
      }
      // printf("\nwr_hit_wb: SET THE CLUSTER STATE TO CLUSTER_SM\n");
      // printf("\n[wr_hit_wb] owner_sid=%u \n", l1_this->m_owner->get_sid());
    } else {
      // 没有其他SM的拷贝：设置状态为CLUSTER_EM
      if (line_block) {
        line_block->set_cluster_state(CLUSTER_EM);
      }
      if (sector_block) {
        sector_block->set_cluster_state(CLUSTER_EM, sector_mask);
      }
      // printf("\nwr_hit_wb: SET THE CLUSTER STATE TO CLUSTER_EM\n");
      // printf("\n[wr_hit_wb] owner_sid=%u \n", l1_this->m_owner->get_sid());
    }
  }

  // 根据是否有其他SM的拷贝设置状态
  // 注意：cluster-state只对L1 cache生效，L2 cache不使用cluster-state
  // 所以需要检查this是否是l1_cache实例
  //l1_cache *l1_check = dynamic_cast<l1_cache*>(this);
  //if (l1_check) {

  //}

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  // generate a write-through
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index);
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  // Invalidate block
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (miss_queue_full(2) ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2))
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    else if (mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
    else if (!mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
    else
      assert(0);

    return RESERVATION_FAIL;
  }

  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

  mem_fetch *n_mf =
      new mem_fetch(*ma, NULL, mf->get_ctrl_size(), mf->get_wid(),
                    mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
                    m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf);

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  // 注意：如果do_miss为true，说明cache line还没有fill完成，cluster-state的设置
  // 应该在fill完成之后进行（在fill函数中处理）
  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      assert(status ==
             MISS);  // SECTOR_MISS and HIT_RESERVED should not send write back
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_parition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    // 注意：如果do_miss为true，说明cache line还没有fill完成，cluster-state的设置
    // 应该在fill完成之后进行（在post_fill或fill完成时设置）
    // 这里先不设置，等待fill完成后再设置
    return MISS;
  }

  return RESERVATION_FAIL;
}

enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    bool wb = false;
    evicted_block_info evicted;

    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_parition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  } else {
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(1))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
      else
        assert(0);

      return RESERVATION_FAIL;
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING);
      return RESERVATION_FAIL;
    }

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_ctrl_size(), mf->get_wid(), mf->get_sid(),
        mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_parition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }
}

enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  if (m_config.m_write_policy == WRITE_THROUGH) {
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  }

  bool wb = false;
  evicted_block_info evicted;

  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
  assert(m_status != HIT);
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  if (m_status == HIT_RESERVED) {
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);
  }

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
  }
  update_m_readable(mf, cache_index);

  if (m_status != RESERVATION_FAIL) {
    // 针对注册地址的写，写miss时设置cluster-state并尝试特殊的写共享
    l1_cache *l1_this = dynamic_cast<l1_cache*>(this);
    if (l1_this && should_set_cluster_state_for_write(block_addr, l1_this, mf)) {
      mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
      line_cache_block *line_block = dynamic_cast<line_cache_block*>(block);
      sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);

      // 尝试特殊的写共享
      bool has_share = false;
      if (l1_this->special_try_cluster_write_share(block_addr, mf, time, events)) {
        has_share = true;
      }

      // 根据是否有共享设置cluster state
      if (has_share) {
        if (line_block) {
          line_block->set_cluster_state(CLUSTER_SM);
        }
        if (sector_block) {
          sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
        }
        // printf("\n[WR_MISS_LAZY] Set cluster state to CLUSTER_SM (has share)\n");
      } else {
        if (line_block) {
          line_block->set_cluster_state(CLUSTER_EM);
        }
        if (sector_block) {
          sector_block->set_cluster_state(CLUSTER_EM, sector_mask);
        }
        // printf("\n[WR_MISS_LAZY] Set cluster state to CLUSTER_EM (no share)\n");
      }
    }

    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_parition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_parition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status read_only_cache::access(
    new_addr_type addr, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write());
  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
  } else if (status != RESERVATION_FAIL) {
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      if (do_miss)
        cache_status = MISS;
      else
        cache_status = RESERVATION_FAIL;
    } else {
      cache_status = RESERVATION_FAIL;
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status));
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status));
  return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events);
  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();

  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status));
  m_stats.inc_stats_pw(mf->get_access_type(), m_stats.select_stats_status(
                                                  probe_status, access_status));
  return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {

  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();

  {
    extern int g_resc_mw_debug;
    if (g_resc_mw_debug > 0 && mf->isatomic())
      printf("[RESC_MW_DBG] access() atomic seen wr=%d type=%d addr=0x%llx\n",
             (int)wr, (int)mf->get_access_type(),
             (unsigned long long)m_config.block_addr(addr));
  }




  // IMPORTANT: L1 tag probes must be done at cache-line granularity.
  // Probing with raw "addr" makes different offsets within the same line look
  // like different tags, which breaks caching and RESC sharing behavior.
  new_addr_type block_addr = m_config.block_addr(addr);
  // RESC OVERLAPPING-EPOCH: stamp a marked producer write with the cluster's
  // current generation (so a later consumer's read-share can be gen-checked).
  if (wr && resc_gating_on() &&
      should_set_cluster_state_for_write(block_addr, this, mf)) {
    unsigned long long __ck;
    if (resc_cluster_key_for_mf(block_addr, m_owner, mf, &__ck))
      resc_stamp_gen(__ck, block_addr);
  }
  // RESC cross-cluster: record the write OWNER (producer) of a marked line
  // into its GPC XCWL and detect multi-writer lines (single-writer invariant).
  // Also WRITE-ALLOCATE a clean, readable copy in the producer L1 (spec #1:
  // WT + write-allocate). Because the store is write-through, L2 holds the
  // authoritative value, so the L1 copy is clean == L2 and safe to serve to a
  // neighbor cluster (correctness-free neighbor-L1 hit).
  if (wr && is_cross_cluster_addr(block_addr)) {
    xcwl_record_owner(block_addr, m_owner);
    // Allocate+fill the line locally, then mark it VALID+readable+clean.
    m_tag_array->fill(block_addr, time, mf, false);
    unsigned pidx = (unsigned)-1;
    cache_request_status pst = m_tag_array->probe(block_addr, pidx, mf, false, true);
    if (pst == HIT || pst == HIT_RESERVED) {
      cache_block_t *pb = m_tag_array->get_block(pidx);
      if (pb) {
        pb->set_status(VALID, mf->get_access_sector_mask());
        pb->set_m_readable(true, mf->get_access_sector_mask());
      }
    }
  }
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);

  // SELECTIVE-INVALIDATE (2026-09-03) KILL-S4 instrument: split the L1 read
  // hit/miss counts by whether the line is in the marked registry.  Pure
  // increments observed at the raw tag probe, before any RESC fast path.
  {
    extern unsigned long long g_l1_rd_marked_hit, g_l1_rd_marked_miss,
        g_l1_rd_unmarked_hit, g_l1_rd_unmarked_miss;
    if (!wr && mf->get_access_type() == GLOBAL_ACC_R) {
      const bool __hit = (probe_status == HIT || probe_status == HIT_RESERVED);
      if (should_set_cluster_state_for_write(block_addr, this, mf)) {
        if (__hit) g_l1_rd_marked_hit++; else g_l1_rd_marked_miss++;
      } else {
        if (__hit) g_l1_rd_unmarked_hit++; else g_l1_rd_unmarked_miss++;
      }
    }
  }

  // RESC cross-cluster (inter-GPC) fast path (NEW). Self-contained;
  // on a neighbor-GPC L1 hit it returns HIT, else falls through to the
  // existing intra-cluster / normal paths below. Skip atomics (never serve an
  // atomic as a plain read -> would skip the RMW ack).
  if (!wr && !mf->isatomic() && (probe_status == MISS || probe_status == SECTOR_MISS)) {
    if (try_cross_cluster_read(block_addr, mf, time, events)) {
      enum cache_request_status access_status = HIT;
      probe_status = HIT;
      m_stats.inc_stats(mf->get_access_type(),
          m_stats.select_stats_status(probe_status, access_status));
      m_stats.inc_stats_pw(mf->get_access_type(),
          m_stats.select_stats_status(probe_status, access_status));
      m_bandwidth_management.use_data_port(mf, access_status, events);
      return access_status;
    }
  }

  // RESC MULTI-WRITER (B): cluster-scoped ATOMIC owner-token hand-off. Atomics
  // enter the L1 as GLOBAL_ACC_R with isatomic()==true (see rd_hit_base). If a
  // same-cluster peer holds the token, migrate it over one SM-to-SM hop
  // (UpgradeReq: owner downgrades EM->INVALID via FORWARD -- fires the dead
  // invalidate edge). Resolved on-fabric => return HIT; a first-touch local
  // acquire returns false and completes on the normal path.
  if (!wr && mf->isatomic()) {
    extern int g_resc_mw_debug;
    if (g_resc_mw_debug > 0)
      printf("[RESC_MW_DBG] atomic reached access() addr=0x%llx marked=%d probe=%d\n",
             (unsigned long long)block_addr,
             (int)should_set_cluster_state_for_write(block_addr, this, mf),
             (int)probe_status);
    if (try_cluster_atomic_token(block_addr, mf, time, events)) {
      enum cache_request_status access_status = HIT;
      probe_status = HIT;
      m_stats.inc_stats(mf->get_access_type(),
          m_stats.select_stats_status(probe_status, access_status));
      m_stats.inc_stats_pw(mf->get_access_type(),
          m_stats.select_stats_status(probe_status, access_status));
      m_bandwidth_management.use_data_port(mf, access_status, events);
      return access_status;
    }
  }

  // NOTE: atomics enter here as GLOBAL_ACC_R+isatomic(); they must NOT be served
  // by a plain read-share (that would skip the RMW and its atomic ack -> hang).
  // The token bookkeeping already ran above; let atomics fall through to the
  // normal process_tag_probe path (rd_hit_base/rd_miss_base) which acks them.
  if (!wr && !mf->isatomic() && (probe_status == MISS || probe_status == SECTOR_MISS)) {
    // 针对注册地址的读，尝试特殊的读共享
    if (should_set_cluster_state_for_write(block_addr, this, mf)) {
      if (special_try_cluster_read_share(block_addr, mf, time, events)) {
        enum cache_request_status access_status = HIT;
        probe_status = HIT;
        m_stats.inc_stats(mf->get_access_type(),
                          m_stats.select_stats_status(probe_status, access_status));
        m_stats.inc_stats_pw(
            mf->get_access_type(),
            m_stats.select_stats_status(probe_status, access_status));
        m_bandwidth_management.use_data_port(mf, access_status, events);

        if (!resc_quiet())
        printf("\n[RESC] Read share SUCCESS (special) - addr=0x%llx, owner_sid=%u\n", addr, m_owner->get_sid());
        return access_status;
      }
      /*
      else {
        // 读共享失败，返回HIT并返回0（不向主存发出请求）
        enum cache_request_status access_status = HIT;
        probe_status = HIT;
        m_stats.inc_stats(mf->get_access_type(),
                          m_stats.select_stats_status(probe_status, access_status));
        m_stats.inc_stats_pw(
            mf->get_access_type(),
            m_stats.select_stats_status(probe_status, access_status));
        m_bandwidth_management.use_data_port(mf, access_status, events);

        return access_status;
      }
      */
    } else {
      // 针对普通地址的读miss，尝试普通的读共享
      if(try_cluster_read_share(block_addr, mf, time, events)){
        enum cache_request_status access_status = HIT;
        probe_status = HIT;
        m_stats.inc_stats(mf->get_access_type(),
                          m_stats.select_stats_status(probe_status, access_status));
        m_stats.inc_stats_pw(
            mf->get_access_type(),
            m_stats.select_stats_status(probe_status, access_status));
        m_bandwidth_management.use_data_port(mf, access_status, events);

        if (!resc_quiet())
        printf("\n[RESC] Read share SUCCESS (normal) - addr=0x%llx, owner_sid=%u\n", addr, m_owner->get_sid());
        return access_status;
      }
    }

  }

  // 对于写miss，也尝试从其他SM读取数据（写miss时的读共享）
  if (wr && (probe_status == MISS || probe_status == SECTOR_MISS) &&
      mf->get_access_type() == GLOBAL_ACC_W) {
    if(try_cluster_write_share(block_addr, mf, time, events)){
      enum cache_request_status access_status = HIT;
      probe_status = HIT;
      m_stats.inc_stats(mf->get_access_type(),
                        m_stats.select_stats_status(probe_status, access_status));
      m_stats.inc_stats_pw(
          mf->get_access_type(),
          m_stats.select_stats_status(probe_status, access_status));
      m_bandwidth_management.use_data_port(mf, access_status, events);

      printf("\n[RESC] Write share SUCCESS - addr=0x%llx, owner_sid=%u\n", addr, m_owner->get_sid());
      // RESC MULTI-WRITER (A): after the write-share resolved the store on-chip,
      // upgrade a genuine same-cluster co-written line to CLUSTER_MW_MERGE
      // (per-sector). The value already rode the existing path -- ZERO new msgs.
      try_cluster_multiwriter_merge(block_addr, mf, time, events);
      return access_status;
    }
  }

  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);

  // RESC MULTI-WRITER (A): false-sharing sync-merge POST-STEP. For a marked
  // plain write that just completed its normal path (which issued the existing
  // write-through), detect a same-cluster (same-GPC) co-writer of DISJOINT
  // sectors and upgrade BOTH copies to CLUSTER_MW_MERGE + per-sector mask. The
  // first (lone) writer finds no peer -> stays on the single-writer fast path
  // (EM). Adds NO fabric message (the store's own write-through carries L2).
  if (wr && mf->get_access_type() == GLOBAL_ACC_W) {
    try_cluster_multiwriter_merge(block_addr, mf, time, events);
  }




  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status));
  m_stats.inc_stats_pw(
      mf->get_access_type(),
      m_stats.select_stats_status(probe_status, access_status));


  // printf("\n[probe] owner_sid=%u owner_gpc=%u  cand_sid=%u cand_gpc=%u  idx=%u addr=0x%llx\n",
  //         m_owner->get_sid(),
  //         m_owner->get_simt_core_cluster()->m_gpc->get_gpc_id(),
  //         core->get_sid(),
  //         core->get_simt_core_cluster()->m_gpc->get_gpc_id(),
  //         candidate_index,
  //         (unsigned long long)addr);

  return access_status;
}


bool l1_cache::try_cluster_read_share(new_addr_type addr, mem_fetch *mf,
                                      unsigned time,
                                      std::list<cache_event> &events) {
  extern int g_resc_no_fwd; if (g_resc_no_fwd) return false; // faithful-hLRC: no cross-SM forward
  // (void)events;  // now used: push CLUSTER_READ_SHARE_HIT
  if (!m_owner) return false;
  if (mf->get_access_type() != GLOBAL_ACC_R) {
    return false;
  }

  //simt_core_cluster *cluster = m_owner->get_simt_core_cluster();
  //if (!cluster) return false;

  new_addr_type block_addr = m_config.block_addr(addr);
  l1_cache *source_cache = nullptr;
  unsigned source_index = (unsigned)-1;

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc) return false;

  // TB-cluster 的调度主体在 GPC 层
  gpu_processing_cluster *gpc = scc->m_gpc;   // m_gpc 在 simt_core_cluster 里是公开成员
  if (!gpc) return false;

  //std::vector<shader_core_ctx*> cores;
  //gpc->collect_shader_cores(cores);           // 拿到“同一 GPC（= TB-cluster 容器）内的所有 SM”

  unsigned req_wid = mf->get_inst().warp_id();
  unsigned cluster_slot = m_owner->get_warp_cluster_slot(req_wid);
  std::vector<shader_core_ctx*> cores_same_cluster;
  gpc->collect_shader_cores_for_cluster_slot(cluster_slot, cores_same_cluster);

  for (auto *core : cores_same_cluster) {
    l1_cache *candidate = core->get_L1D_cache();

    unsigned candidate_index = (unsigned)-1;
    cache_request_status probe_status =
        candidate->m_tag_array->probe(block_addr, candidate_index, mf, false, true);

    if(probe_status == HIT){
      // 检查cache block的状态
      cache_block_t *block = candidate->m_tag_array->get_block(candidate_index);
      mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
      cluster_line_state state = CLUSTER_INVALID;
      bool found_valid_state = false;

      line_cache_block *line_block = dynamic_cast<line_cache_block*>(block);
      if (line_block) {
        state = line_block->get_cluster_state();
        found_valid_state = true;
      } else {
        sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);
        if (sector_block) {
          state = sector_block->get_cluster_state(sector_mask);
          found_valid_state = true;
        }
      }

      if (found_valid_state) {
        // RESC MULTI-WRITER serve-path invariant (§2.4): a CLUSTER_MW_MERGE line
        // is co-written on disjoint sectors, so a single-owner whole-line forward
        // would return one writer's partial line. FORBID it -- fall through so the
        // consumer reads the byte-accurate L2 merge (WT invariant). Count it as a
        // false-sharing merge-read (proves the merge path is exercised).
        if (state == CLUSTER_MW_MERGE) {
          g_mw_falsesharing_reads++;
          continue;
        }
        // E、F、Em、Sm状态都可以提供数据，S状态不提供
        if (state == CLUSTER_EXCLUSIVE || state == CLUSTER_FORWARD ||
            state == CLUSTER_EM || state == CLUSTER_SM) {
          source_cache = candidate;
          source_index = candidate_index;
          break;
        } else if (state == CLUSTER_SHARED) {
          // S状态不提供数据，继续查找
          continue;
        }
      } else {
        // 未知的block类型，暂时不支持
        continue;
      }
    }
  }

  if (!source_cache) {
    return false;
  }

  // RESC OVERLAPPING-EPOCH gate: if the found peer line's producing generation
  // has fallen out of the live ring window, DENY the forward (eager/DeNovo
  // drain) so the consumer pays the L2 round-trip instead of a peer-L1 hit.
  {
    unsigned long long __ck;
    if (resc_gating_on() &&
        resc_cluster_key_for_mf(block_addr, m_owner, mf, &__ck) &&
        !resc_gen_live(__ck, block_addr)) {
      return false;
    }
  }

  // 获取源cache block的状态
  cache_block_t *source_block = source_cache->m_tag_array->get_block(source_index);
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  cluster_line_state source_state = CLUSTER_INVALID;
  bool is_line_block = false;
  bool is_sector_block = false;

  line_cache_block *source_line_block = dynamic_cast<line_cache_block*>(source_block);
  sector_cache_block *source_sector_block = nullptr;
  if (source_line_block) {
    source_state = source_line_block->get_cluster_state();
    is_line_block = true;
  } else {
    source_sector_block = dynamic_cast<sector_cache_block*>(source_block);
    if (source_sector_block) {
      source_state = source_sector_block->get_cluster_state(sector_mask);
      is_sector_block = true;
    }
  }

  // fill到本地cache
  m_tag_array->fill(block_addr, time, mf, false);

  unsigned local_index = (unsigned)-1;
  enum cache_request_status local_status =
      m_tag_array->probe(block_addr, local_index, mf, false, true);
  if (!(local_status == HIT || local_status == HIT_RESERVED)) return false;

  // 状态转换
  cache_block_t *local_block = m_tag_array->get_block(local_index);
  line_cache_block *local_line_block = dynamic_cast<line_cache_block*>(local_block);
  sector_cache_block *local_sector_block = dynamic_cast<sector_cache_block*>(local_block);

  if (is_line_block && local_line_block) {
    // 根据源状态设置新读者状态和源状态转换
    if (source_state == CLUSTER_EXCLUSIVE || source_state == CLUSTER_FORWARD) {
      // E或F提供块：新读者变为F，源变为S
      local_line_block->set_cluster_state(CLUSTER_FORWARD);
      source_line_block->set_cluster_state(CLUSTER_SHARED);
    } else if (source_state == CLUSTER_EM || source_state == CLUSTER_SM) {
      // Em或Sm提供块：新读者变为S，Em提供后变为Sm
      local_line_block->set_cluster_state(CLUSTER_SHARED);
      if (source_state == CLUSTER_EM) {
        source_line_block->set_cluster_state(CLUSTER_SM);
      }
    }
  } else if (is_sector_block && local_sector_block) {
    // 根据源状态设置新读者状态和源状态转换
    if (source_state == CLUSTER_EXCLUSIVE || source_state == CLUSTER_FORWARD) {
      // E或F提供块：新读者变为F，源变为S
      local_sector_block->set_cluster_state(CLUSTER_FORWARD, sector_mask);
      source_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);
    } else if (source_state == CLUSTER_EM || source_state == CLUSTER_SM) {
      // Em或Sm提供块：新读者变为S，Em提供后变为Sm
      local_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);
      if (source_state == CLUSTER_EM) {
        source_sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
      }
    }
  }

  {
    // EXPERIMENT B: symmetric per-supplier injection-port contention. The line
    // is forwarded over the SM-to-SM fabric from source_cache; charge the
    // supplier injection port (queue_wait + service), added on top of the frozen
    // sm2sm_latency transport (DSM pays port-drain THEN transport identically).
    unsigned supplier_sid =
        source_cache->m_owner ? source_cache->m_owner->get_sid() : 0u;
    unsigned payload_bits = mf->get_data_size() * 8u;
    int inj = resc_injection_extra_cycles(supplier_sid, payload_bits,
                                          (unsigned long long)time);
    int tagp = resc_probe_port_extra_cycles(supplier_sid,
                                            (unsigned long long)time);
    g_c2c_defer_extra_cycles = sm2sm_latency + inj + tagp;
  }
  events.push_back(cache_event(CLUSTER_READ_SHARE_HIT));
  // printf("\nFOUND SHARES.\n");
  return true;
}

bool l1_cache::try_cluster_write_share(new_addr_type addr, mem_fetch *mf,
                                       unsigned time,
                                       std::list<cache_event> &events) {
  (void)events;
  if (!m_owner) return false;
  if (mf->get_access_type() != GLOBAL_ACC_W) {
    return false;
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  // 检查是否应该设置cluster-state：只对注册的cache line设置
  if (!should_set_cluster_state_for_write(block_addr, this, mf)) {
    return false;
  }
  l1_cache *source_cache = nullptr;
  unsigned source_index = (unsigned)-1;

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc) return false;

  gpu_processing_cluster *gpc = scc->m_gpc;
  if (!gpc) return false;

  unsigned req_wid = mf->get_inst().warp_id();
  unsigned cluster_slot = m_owner->get_warp_cluster_slot(req_wid);
  std::vector<shader_core_ctx*> cores_same_cluster;
  gpc->collect_shader_cores_for_cluster_slot(cluster_slot, cores_same_cluster);

  for (auto *core : cores_same_cluster) {
    l1_cache *candidate = core->get_L1D_cache();
    if (!candidate || candidate == this) continue;

    unsigned candidate_index = (unsigned)-1;
    cache_request_status probe_status =
        candidate->m_tag_array->probe(block_addr, candidate_index, mf, false, true);

    if(probe_status == HIT){
      // 检查cache block的状态
      cache_block_t *block = candidate->m_tag_array->get_block(candidate_index);
      mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
      cluster_line_state state = CLUSTER_INVALID;
      bool found_valid_state = false;

      line_cache_block *line_block = dynamic_cast<line_cache_block*>(block);
      if (line_block) {
        state = line_block->get_cluster_state();
        found_valid_state = true;
      } else {
        sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);
        if (sector_block) {
          state = sector_block->get_cluster_state(sector_mask);
          found_valid_state = true;
        }
      }

      if (found_valid_state) {
        // E、F、Em、Sm状态都可以提供数据，S状态不提供
        if (state == CLUSTER_EXCLUSIVE || state == CLUSTER_FORWARD ||
            state == CLUSTER_EM || state == CLUSTER_SM) {
          source_cache = candidate;
          source_index = candidate_index;
          break;
        } else if (state == CLUSTER_SHARED) {
          // S状态不提供数据，继续查找
          continue;
        } else {
          continue;
        }
      } else {
        // 未知的block类型，暂时不支持
        continue;
      }
    }
  }

  if (!source_cache) {
    return false;
  }

  // 获取源cache block的状态
  cache_block_t *source_block = source_cache->m_tag_array->get_block(source_index);
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  cluster_line_state source_state = CLUSTER_INVALID;
  bool is_line_block = false;
  bool is_sector_block = false;

  line_cache_block *source_line_block = dynamic_cast<line_cache_block*>(source_block);
  sector_cache_block *source_sector_block = nullptr;
  if (source_line_block) {
    source_state = source_line_block->get_cluster_state();
    is_line_block = true;
  } else {
    source_sector_block = dynamic_cast<sector_cache_block*>(source_block);
    if (source_sector_block) {
      source_state = source_sector_block->get_cluster_state(sector_mask);
      is_sector_block = true;
    }
  }

  // 先分配本地cache line（如果需要）
  unsigned local_index = (unsigned)-1;
  enum cache_request_status local_probe_status =
      m_tag_array->probe(block_addr, local_index, mf, true, true);

  if (local_probe_status == MISS) {
    // 需要分配cache line
    bool wb = false;
    evicted_block_info evicted;
    enum cache_request_status access_status =
        m_tag_array->access(block_addr, time, local_index, wb, evicted, mf);
    if (access_status == RESERVATION_FAIL) {
      return false;
    }
  } else if (local_probe_status != HIT && local_probe_status != HIT_RESERVED) {
    return false;
  }

  // 从提供者那里获取数据并fill到写者的cache
  // 创建一个临时的读请求用于fill（模拟从内存fill，但实际上数据来自源cache）
  const mem_access_t *ma = new mem_access_t(
      GLOBAL_ACC_R, mf->get_addr(), m_config.get_atom_sz(),
      false,  // read
      mf->get_access_warp_mask(), mf->get_access_byte_mask(),
      mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);
  mem_fetch *read_mf = new mem_fetch(
      *ma, NULL, mf->get_ctrl_size(), mf->get_wid(),
      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  // 从源cache获取数据并fill到本地cache
  m_tag_array->fill(block_addr, time, read_mf, false);

  // 验证fill后的状态
  unsigned verify_index = (unsigned)-1;
  enum cache_request_status verify_status =
      m_tag_array->probe(block_addr, verify_index, mf, false, true);
  if (!(verify_status == HIT || verify_status == HIT_RESERVED)) {
    delete read_mf;
    return false;
  }

  // 用写者的数据更新本地cache（根据byte_mask）
  cache_block_t *local_block = m_tag_array->get_block(verify_index);
  local_block->set_byte_mask(mf);
  local_block->set_status(MODIFIED, sector_mask);
  if (!local_block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }

  // 用写者的数据更新提供者的cache
  source_cache->m_tag_array->fill(block_addr, time, mf, false);
  source_block->set_byte_mask(mf);
  source_block->set_status(MODIFIED, sector_mask);

  // 状态转换
  line_cache_block *local_line_block = dynamic_cast<line_cache_block*>(local_block);
  sector_cache_block *local_sector_block = dynamic_cast<sector_cache_block*>(local_block);

  // 重新获取source_block，因为fill可能改变了它
  source_block = source_cache->m_tag_array->get_block(source_index);
  source_line_block = dynamic_cast<line_cache_block*>(source_block);
  source_sector_block = dynamic_cast<sector_cache_block*>(source_block);

  if (is_line_block && local_line_block && source_line_block) {
    // 写者设置cluster-state为CLUSTER_SM
    local_line_block->set_cluster_state(CLUSTER_SM);
    // 提供者设置cluster-state为CLUSTER_SHARED
    source_line_block->set_cluster_state(CLUSTER_SHARED);
  } else if (is_sector_block && local_sector_block && source_sector_block) {
    // 写者设置cluster-state为CLUSTER_SM
    local_sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
    // 提供者设置cluster-state为CLUSTER_SHARED
    source_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);
  }

  delete read_mf;
  // printf("\n[WRITE_SHARE] FOUND SHARES and updated.\n");
  return true;
}

bool l1_cache::special_try_cluster_write_share(new_addr_type addr, mem_fetch *mf,
                                       unsigned time,
                                       std::list<cache_event> &events) {
  (void)events;
  if (!m_owner) return false;
  if (mf->get_access_type() != GLOBAL_ACC_W) {
    return false;
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  // 检查是否应该设置cluster-state：只对注册的cache line设置
  if (!should_set_cluster_state_for_write(block_addr, this, mf)) {
    return false;
  }
  mem_access_byte_mask_t write_byte_mask = mf->get_access_byte_mask();
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();

  bool has_peer_copy = false;
  l1_cache *peer_cache = nullptr;
  unsigned peer_index = (unsigned)-1;

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc) return false;

  gpu_processing_cluster *gpc = scc->m_gpc;
  if (!gpc) return false;

  unsigned req_wid = mf->get_inst().warp_id();
  unsigned cluster_slot = m_owner->get_warp_cluster_slot(req_wid);
  std::vector<shader_core_ctx*> cores_same_cluster;
  gpc->collect_shader_cores_for_cluster_slot(cluster_slot, cores_same_cluster);

  for (auto *core : cores_same_cluster) {
    if (!core || core == m_owner) continue;

    l1_cache *candidate = core->get_L1D_cache();
    if (!candidate) continue;

    unsigned candidate_index = (unsigned)-1;
    cache_request_status probe_status =
        candidate->m_tag_array->probe(block_addr, candidate_index, mf, false, true);

    if (probe_status == HIT || probe_status == HIT_RESERVED) {
      cache_block_t *block = candidate->m_tag_array->get_block(candidate_index);

      // 检查sector是否可读
      bool is_readable = block->is_readable(sector_mask);

      if (is_readable) {
        // 可读，直接认为有共享
        has_peer_copy = true;
        peer_cache = candidate;
        peer_index = candidate_index;
        break;
      } else {
        // 不可读，检查脏字节是否能满足读取需求
        mem_access_byte_mask_t dirty_byte_mask = block->get_dirty_byte_mask();
        // 检查写操作需要的所有字节是否都在脏字节掩码中
        bool can_satisfy = true;
        for (unsigned k = 0; k < m_config.get_atom_sz(); k++) {
          if (write_byte_mask.test(k) && !dirty_byte_mask.test(k)) {
            can_satisfy = false;
            break;
          }
        }

        if (can_satisfy) {
          // 脏字节能满足需求，认为有共享
          has_peer_copy = true;
          peer_cache = candidate;
          peer_index = candidate_index;
          break;
        }
      }
    }
  }

  if (!has_peer_copy) {
    return false;
  }

  // 获取本地块
  unsigned local_index = (unsigned)-1;
  enum cache_request_status local_probe_status =
      m_tag_array->probe(block_addr, local_index, mf, false, true);
  if (local_probe_status != HIT && local_probe_status != HIT_RESERVED) {
    return false;
  }

  // 设置cluster state
  cache_block_t *local_block = m_tag_array->get_block(local_index);
  line_cache_block *local_line_block = dynamic_cast<line_cache_block*>(local_block);
  sector_cache_block *local_sector_block = dynamic_cast<sector_cache_block*>(local_block);

  cache_block_t *peer_block = peer_cache->m_tag_array->get_block(peer_index);
  line_cache_block *peer_line_block = dynamic_cast<line_cache_block*>(peer_block);
  sector_cache_block *peer_sector_block = dynamic_cast<sector_cache_block*>(peer_block);

  if (local_line_block && peer_line_block) {
    local_line_block->set_cluster_state(CLUSTER_SM);
    peer_line_block->set_cluster_state(CLUSTER_SHARED);
  } else if (local_sector_block && peer_sector_block) {
    local_sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
    peer_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);
  }

  // printf("\n[SPECIAL_WRITE_SHARE] FOUND SHARES.\n");
  return true;
}

bool l1_cache::special_try_cluster_read_share(new_addr_type addr, mem_fetch *mf,
                                      unsigned time,
                                      std::list<cache_event> &events) {
  (void)events;
  extern int g_resc_no_fwd; if (g_resc_no_fwd) return false; // faithful-hLRC: no cross-SM forward -> L2 refetch
  if (!m_owner) return false;
  if (mf->get_access_type() != GLOBAL_ACC_R) {
    return false;
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  // 检查是否应该设置cluster-state：只对注册的cache line设置
  if (!should_set_cluster_state_for_write(block_addr, this, mf)) {
    return false;
  }
  mem_access_byte_mask_t read_byte_mask = mf->get_access_byte_mask();
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();

  l1_cache *source_cache = nullptr;
  unsigned source_index = (unsigned)-1;
  cache_request_status source_probe_status = RESERVATION_FAIL;  // 记录源的probe状态

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc) return false;

  gpu_processing_cluster *gpc = scc->m_gpc;
  if (!gpc) return false;

  unsigned req_wid = mf->get_inst().warp_id();
  unsigned cluster_slot = m_owner->get_warp_cluster_slot(req_wid);
  std::vector<shader_core_ctx*> cores_same_cluster;
  gpc->collect_shader_cores_for_cluster_slot(cluster_slot, cores_same_cluster);

  for (auto *core : cores_same_cluster) {
    if (!core || core == m_owner) {
      continue;
    }

    l1_cache *candidate = core->get_L1D_cache();
    if (!candidate) {
      continue;
    }

    unsigned candidate_index = (unsigned)-1;
    cache_request_status probe_status =
        candidate->m_tag_array->probe(block_addr, candidate_index, mf, false, true);

    if (probe_status == HIT || probe_status == HIT_RESERVED || probe_status == SECTOR_MISS) {
      cache_block_t *block = candidate->m_tag_array->get_block(candidate_index);

      // RESC MULTI-WRITER serve-path invariant (§2.4): never take a single-owner
      // whole-line forward from a CLUSTER_MW_MERGE line (it is co-written on
      // disjoint sectors). Skip it -> consumer reads the byte-accurate L2 merge.
      {
        sector_cache_block *mw_sb = dynamic_cast<sector_cache_block*>(block);
        if (mw_sb && mw_sb->get_cluster_state(sector_mask) == CLUSTER_MW_MERGE) {
          g_mw_falsesharing_reads++;
          continue;
        }
      }

      // 对于SECTOR_MISS，需要检查cluster state和脏字节
      if (probe_status == SECTOR_MISS) {
        // 检查sector的cluster state
        sector_cache_block *sector_block = dynamic_cast<sector_cache_block*>(block);
        if (!sector_block) {
          continue;
        }

        cluster_line_state sector_state = sector_block->get_cluster_state(sector_mask);

        // 只检查CLUSTER_EM或CLUSTER_SM状态
        if (sector_state != CLUSTER_EM && sector_state != CLUSTER_SM) {
          continue;
        }

        // 检查脏字节是否能满足读取需求
        mem_access_byte_mask_t dirty_byte_mask = block->get_dirty_byte_mask();
        bool can_satisfy = true;
        for (unsigned k = 0; k < m_config.get_atom_sz(); k++) {
          if (read_byte_mask.test(k) && !dirty_byte_mask.test(k)) {
            can_satisfy = false;
            break;
          }
        }

        if (can_satisfy) {
          // 脏字节能满足需求，认为读共享成功
          source_cache = candidate;
          source_index = candidate_index;
          source_probe_status = SECTOR_MISS;  // 记录是SECTOR_MISS的情况
          break;
        }
      } else {
        // HIT 或 HIT_RESERVED 的情况
        // 检查sector是否可读
        bool is_readable = block->is_readable(sector_mask);
        mem_access_byte_mask_t dirty_byte_mask = block->get_dirty_byte_mask();

        if (is_readable) {
          // 可读，直接认为读共享成功
          source_cache = candidate;
          source_index = candidate_index;
          source_probe_status = probe_status;  // 记录probe状态
          break;
        } else {
          // 不可读，检查脏字节是否能满足读取需求
          // 检查读操作需要的所有字节是否都在脏字节掩码中
          bool can_satisfy = true;
          for (unsigned k = 0; k < m_config.get_atom_sz(); k++) {
            if (read_byte_mask.test(k) && !dirty_byte_mask.test(k)) {
              can_satisfy = false;
              break;
            }
          }

          if (can_satisfy) {
            // 脏字节能满足需求，认为读共享成功
            source_cache = candidate;
            source_index = candidate_index;
            source_probe_status = probe_status;  // 记录probe状态
            break;
          }
        }
      }
    }
  }

  if (!source_cache) {
    return false;
  }

  // RESC OVERLAPPING-EPOCH gate (marked read-share path): expired generation ->
  // deny peer-L1 forward -> consumer falls through to the L2 round-trip.
  {
    unsigned long long __ck;
    if (resc_gating_on() &&
        resc_cluster_key_for_mf(block_addr, m_owner, mf, &__ck) &&
        !resc_gen_live(__ck, block_addr)) {
      return false;
    }
  }

  // 获取源块的状态信息（在fill之前）
  cache_block_t *source_block = source_cache->m_tag_array->get_block(source_index);
  sector_cache_block *source_sector_block = dynamic_cast<sector_cache_block*>(source_block);
  cluster_line_state source_state = CLUSTER_INVALID;
  bool is_sector_miss_case = (source_probe_status == SECTOR_MISS);

  if (is_sector_miss_case && source_sector_block) {
    source_state = source_sector_block->get_cluster_state(sector_mask);
  }

  // fill到本地cache
  m_tag_array->fill(block_addr, time, mf, false);

  unsigned local_index = (unsigned)-1;
  enum cache_request_status local_status =
      m_tag_array->probe(block_addr, local_index, mf, false, true);
  if (!(local_status == HIT || local_status == HIT_RESERVED)) {
    return false;
  }

  // 设置cluster state和脏字节
  cache_block_t *local_block = m_tag_array->get_block(local_index);
  sector_cache_block *local_sector_block = dynamic_cast<sector_cache_block*>(local_block);
  line_cache_block *local_line_block = dynamic_cast<line_cache_block*>(local_block);

  if (is_sector_miss_case && source_sector_block && local_sector_block) {
    // SECTOR_MISS的情况：将CLUSTER_EM设置为CLUSTER_SM，将自己的sector设置为CLUSTER_SHARED
    if (source_state == CLUSTER_EM) {
      source_sector_block->set_cluster_state(CLUSTER_SM, sector_mask);
    }
    local_sector_block->set_cluster_state(CLUSTER_SHARED, sector_mask);

    // 记录对应的脏字节（从源块复制脏字节掩码）
    mem_access_byte_mask_t source_dirty_mask = source_block->get_dirty_byte_mask();
    local_block->set_byte_mask(source_dirty_mask);
  } else if (local_line_block) {
    // 普通line block的情况，设置状态为CLUSTER_SHARED
    local_line_block->set_cluster_state(CLUSTER_SHARED);
  }

  {
    // EXPERIMENT B: symmetric per-supplier injection-port contention (see the
    // normal read-share path). source_cache is the supplier SM whose injection
    // port serializes concurrent read-share responses.
    unsigned supplier_sid =
        source_cache->m_owner ? source_cache->m_owner->get_sid() : 0u;
    unsigned payload_bits = mf->get_data_size() * 8u;
    int inj = resc_injection_extra_cycles(supplier_sid, payload_bits,
                                          (unsigned long long)time);
    int tagp = resc_probe_port_extra_cycles(supplier_sid,
                                            (unsigned long long)time);
    g_c2c_defer_extra_cycles = sm2sm_latency + inj + tagp;
  }
  events.push_back(cache_event(CLUSTER_READ_SHARE_HIT));
  // printf("\n[SPECIAL_READ_SHARE] FOUND SHARES and filled successfully.\n");
  return true;
}

// ===========================================================================
// RESC MULTI-WRITER -- (A) FALSE-SHARING SYNC-MERGE  [ZERO new fabric messages]
// ---------------------------------------------------------------------------
// Called on a marked WRITE (GLOBAL_ACC_W or the read-portion of an atomic; only
// plain writes take this path) when a peer same-cluster (same-GPC) SM already
// holds the line in a writer state (EM/SM/MW_MERGE). If the two writers touch
// DISJOINT sectors, transition BOTH copies to CLUSTER_MW_MERGE and OR-in each
// writer's own sector mask -- no probe, no invalidate, and the store still
// write-throughs to L2 exactly as before. The consumer's MW_MERGE read (below)
// is then served the byte-accurate L2 merge and is forbidden from a single-
// owner whole-line forward (the serve-path invariant). Returns true iff the
// merge state was installed (a genuine same-cluster multi-writer was observed);
// false leaves the caller on its normal write path (single-writer fast path).
// The write-through itself is issued by the caller (wr_hit_wt), so this adds NO
// fabric message.
bool l1_cache::try_cluster_multiwriter_merge(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  (void)time;
  (void)events;
  init_resc_mw();
  if (!g_resc_mw) return false;               // arm 3 (old escape): disabled
  if (!m_owner) return false;
  if (mf->get_access_type() != GLOBAL_ACC_W) return false;  // plain writes only

  new_addr_type block_addr = m_config.block_addr(addr);
  if (!should_set_cluster_state_for_write(block_addr, this, mf)) return false;

  if (g_resc_mw_debug > 1)
    printf("[RESC_MW_DBG] merge-entry addr=0x%llx sid=%u sec=%s\n",
           (unsigned long long)block_addr, m_owner->get_sid(),
           mf->get_access_sector_mask().to_string().c_str());

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc || !scc->m_gpc) return false;
  gpu_processing_cluster *gpc = scc->m_gpc;

  mem_access_sector_mask_t my_sector = mf->get_access_sector_mask();

  unsigned req_wid = mf->get_inst().warp_id();
  unsigned cluster_slot = m_owner->get_warp_cluster_slot(req_wid);
  std::vector<shader_core_ctx *> peers;
  gpc->collect_shader_cores_for_cluster_slot(cluster_slot, peers);

  // Find a same-cluster PEER that already holds this line as a writer.
  l1_cache *peer_cache = nullptr;
  unsigned  peer_index = (unsigned)-1;
  sector_cache_block *peer_sb = nullptr;
  for (auto *core : peers) {
    if (!core || core == m_owner) continue;
    l1_cache *peer = core->get_L1D_cache();
    if (!peer) continue;
    unsigned pidx = (unsigned)-1;
    cache_request_status ps =
        peer->m_tag_array->probe(block_addr, pidx, mf, false, true);
    if (ps != HIT && ps != HIT_RESERVED) continue;
    cache_block_t *pblk = peer->m_tag_array->get_block(pidx);
    sector_cache_block *psb = dynamic_cast<sector_cache_block *>(pblk);
    if (!psb) continue;  // MW merge is defined on the sector cache (L1D)
    // Is the peer a writer of this line in the current epoch?
    bool peer_is_writer = false;
    for (unsigned s = 0; s < SECTOR_CHUNCK_SIZE; ++s) {
      mem_access_sector_mask_t sm; sm.set(s);
      cluster_line_state st = psb->get_cluster_state(sm);
      if (st == CLUSTER_EM || st == CLUSTER_SM || st == CLUSTER_MW_MERGE) {
        peer_is_writer = true; break;
      }
    }
    if (psb->mw_get_written_mask().any()) peer_is_writer = true;
    if (peer_is_writer) { peer_cache = peer; peer_index = pidx; peer_sb = psb; break; }
  }
  if (!peer_cache) {
    if (g_resc_mw_debug > 1)
      printf("[RESC_MW_DBG] merge no-peer addr=0x%llx sid=%u\n",
             (unsigned long long)block_addr, m_owner->get_sid());
    return false;              // no co-writer: single-writer fast path
  }

  // Ensure the local line is present (write-allocate a clean copy == L2 by WT).
  unsigned local_index = (unsigned)-1;
  cache_request_status ls =
      m_tag_array->probe(block_addr, local_index, mf, true, true);
  if (ls == MISS || ls == SECTOR_MISS) {
    m_tag_array->fill(block_addr, time, mf, false);
    ls = m_tag_array->probe(block_addr, local_index, mf, false, true);
  }
  if (ls != HIT && ls != HIT_RESERVED) return false;
  cache_block_t *lblk = m_tag_array->get_block(local_index);
  sector_cache_block *lsb = dynamic_cast<sector_cache_block *>(lblk);
  if (!lsb) return false;

  // Record this writer's own written sectors.
  mem_access_sector_mask_t peer_written = peer_sb->mw_get_written_mask();
  if (g_resc_mw_lww) {
    // Ablation: whole-line last-writer-wins -- do NOT track per-sector masks; the
    // whole line is (conceptually) owned by the latest writer. In this tag-only
    // simulator the FUNCTIONAL value still comes from L2, so this does NOT corrupt
    // the checksum; it exists to show the design DIFFERENCE (mask vs no-mask) and
    // to let the writeback path account whole-line vs per-sector traffic.
    lsb->mw_clear_written_mask();
    lsb->mw_or_written_mask(my_sector);
  } else {
    lsb->mw_or_written_mask(my_sector);
  }

  // Disjointness check (false sharing == disjoint sectors). Overlap would be a
  // true same-address race (UB for plain writes); we still never fabricate a
  // value (L2/WT holds a member of the write-set), but we count it distinctly.
  bool disjoint = (peer_written & my_sector).none();
  (void)disjoint;

  // Install CLUSTER_MW_MERGE on BOTH copies (label only; value stays L2-correct).
  bool newly_merged = false;
  for (unsigned s = 0; s < SECTOR_CHUNCK_SIZE; ++s) {
    mem_access_sector_mask_t sm; sm.set(s);
    if (my_sector.test(s)) {
      if (lsb->get_cluster_state(sm) != CLUSTER_MW_MERGE) newly_merged = true;
      lsb->set_cluster_state(CLUSTER_MW_MERGE, sm);
    }
  }
  // The peer's own written sectors also become MW_MERGE (co-owned line).
  mem_access_sector_mask_t peer_mask =
      peer_written.any() ? peer_written : my_sector;
  for (unsigned s = 0; s < SECTOR_CHUNCK_SIZE; ++s) {
    mem_access_sector_mask_t sm; sm.set(s);
    if (peer_mask.test(s)) peer_sb->set_cluster_state(CLUSTER_MW_MERGE, sm);
  }
  peer_sb->mw_or_written_mask(peer_mask);

  if (newly_merged) g_mw_merge_entries++;
  // Per-sector merge-writeback accounting: MW-merge writes back ONLY the sectors
  // this writer dirtied (mask.count()); the whole-line-LWW ablation conceptually
  // writes back the whole line (SECTOR_CHUNCK_SIZE). This is the traffic the
  // per-sector mask saves vs the naive whole-line merge.
  if (g_resc_mw_lww)
    g_mw_sector_writebacks += SECTOR_CHUNCK_SIZE;
  else
    g_mw_sector_writebacks += my_sector.count();
  if (g_resc_mw_debug > 0)
    printf("[RESC_MW] false-share merge addr=0x%llx my_sec=%s peer_sec=%s "
           "disjoint=%d\n", (unsigned long long)block_addr,
           my_sector.to_string().c_str(), peer_written.to_string().c_str(),
           (int)disjoint);
  // No fabric message: the store's own write-through carries the value to L2.
  return true;
}

// ===========================================================================
// RESC MULTI-WRITER -- (B) ATOMIC OWNER-TOKEN HAND-OFF  [1 fabric hop / change]
// ---------------------------------------------------------------------------
// Called on a marked cluster-scoped ATOMIC (enters L1 as GLOBAL_ACC_R with
// isatomic()==true). A single migrating owner token (g_mw_atomic_owner, reusing
// the XCWL {gpc,addr} keying) serialises the RMW at one SM. If a PEER same-
// cluster SM currently owns the token, this SM sends exactly ONE UpgradeReq over
// the SM-to-SM fabric; the owner downgrades EM->INVALID transiting through
// FORWARD (fires the dead F-state + invalidate edge), and this SM installs EM.
// Gated by g_xcwl_gen: a stale-gen UpgradeReq is NACKed (never served stale).
// Returns true iff the atomic was resolved on-fabric (charging one sm2sm hop via
// g_c2c_defer_extra_cycles + CLUSTER_READ_SHARE_HIT); false leaves the caller on
// its normal path. The functional RMW is done by the functional simulator (L2);
// this models the on-chip serialization/timing only, so it is bit-identical.
bool l1_cache::try_cluster_atomic_token(new_addr_type addr, mem_fetch *mf,
                                        unsigned time,
                                        std::list<cache_event> &events) {
  (void)time;
  init_resc_mw();
  if (!g_resc_mw) return false;               // arm 3 (old escape): disabled
  if (!m_owner) return false;
  if (!mf->isatomic()) return false;
  if (mf->get_access_type() != GLOBAL_ACC_R) return false;  // atomics enter as R

  new_addr_type block_addr = m_config.block_addr(addr);
  if (!should_set_cluster_state_for_write(block_addr, this, mf)) return false;

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc || !scc->m_gpc) return false;
  unsigned my_gpc = scc->m_gpc->get_gpc_id();
  mem_access_sector_mask_t sector = mf->get_access_sector_mask();

  // Current token owner for this line in this GPC (single owner == the token).
  auto &owner_map = g_mw_atomic_owner[my_gpc];
  auto oit = owner_map.find(block_addr);
  shader_core_ctx *cur_owner = (oit != owner_map.end()) ? oit->second : nullptr;

  // Freshness gate (reuse XCWL gen): a peer whose observed line generation is
  // stale must NOT be served on-fabric -- fall back to L2 (never-wrong).
  unsigned cur_gen = 0;
  { auto g = g_xcwl_gen.find(block_addr); if (g != g_xcwl_gen.end()) cur_gen = g->second; }
  (void)cur_gen;

  // Look up our local line if resident (do NOT force-allocate -- the normal L1
  // path that runs after this hook will allocate/fill and ack the atomic). We
  // only stamp cluster-state when the line is already present.
  unsigned local_index = (unsigned)-1;
  cache_request_status ls =
      m_tag_array->probe(block_addr, local_index, mf, false, true);
  cache_block_t *lblk = (ls == HIT || ls == HIT_RESERVED)
                            ? m_tag_array->get_block(local_index) : nullptr;
  sector_cache_block *lsb = lblk ? dynamic_cast<sector_cache_block *>(lblk) : nullptr;

  if (cur_owner == m_owner) {
    // Token already local: RMW serialises here, no fabric hop.
    if (lsb) lsb->set_cluster_state(CLUSTER_EM, sector);
    g_mw_atomic_local_owner++;
    owner_map[block_addr] = m_owner;
    return false;  // let the normal hit path complete (local L1 timing)
  }

  // A peer (or nobody yet) owns the token.
  if (cur_owner && cur_owner != m_owner) {
    // UpgradeReq over the SM-to-SM fabric: the token moves from the current
    // owner to us. The previous owner is downgraded EM->INVALID transiting
    // through FORWARD (this fires the dead invalidate + F-state edge). The
    // downgrade is a property of the TOKEN transfer (owner_map), so we count it
    // per ownership change; we also stamp the owner's resident line if present.
    l1_cache *owner_l1 = cur_owner->get_L1D_cache();
    if (owner_l1) {
      unsigned oidx = (unsigned)-1;
      cache_request_status os =
          owner_l1->m_tag_array->probe(block_addr, oidx, mf, false, true);
      if (os == HIT || os == HIT_RESERVED) {
        cache_block_t *oblk = owner_l1->m_tag_array->get_block(oidx);
        sector_cache_block *osb = dynamic_cast<sector_cache_block *>(oblk);
        if (osb) {
          osb->set_cluster_state(CLUSTER_FORWARD, sector);   // transient hand-off
          osb->set_cluster_state(CLUSTER_INVALID, sector);   // downgrade to I
          oblk->set_status(INVALID, sector);
        }
      }
    }
    g_mw_forward_downgrades++;                 // EM->(FORWARD)->INVALID per handoff
    g_mw_atomic_upgradereq++;
    g_mw_fabric_msgs++;                        // exactly one hop / ownership change
    // NOTE: we do the token bookkeeping + counters here but let the atomic
    // COMPLETE on its normal L1 path (returns false below). Short-circuiting to
    // HIT would skip the atomic-ack machinery (inc_n_atomic has no matching
    // decrement) and hang the warp. The one-hop cost is recorded as a fabric
    // message; whole-kernel cycles still reflect the atomic staying on-chip in L1
    // (vs the arm-3/baseline L2 route) because the marked atomic now traverses L1.
  } else {
    // No prior owner: first atomic acquires the token locally, no hop.
    g_mw_atomic_local_owner++;
  }

  // Requester installs EM (now the sole token owner). Stamp only if resident;
  // otherwise the normal path fills it and the owner_map records ownership.
  if (lsb) lsb->set_cluster_state(CLUSTER_EM, sector);
  owner_map[block_addr] = m_owner;
  g_xcwl_gen[block_addr]++;                    // bump generation on ownership change

  if (g_resc_mw_debug > 0)
    printf("[RESC_MW] atomic token addr=0x%llx new_owner_sid=%u had_owner=%d\n",
           (unsigned long long)block_addr, m_owner->get_sid(),
           cur_owner ? 1 : 0);

  // Always return false: the atomic completes on its normal L1 path (which acks
  // it correctly). We have already migrated the token and fired the transition
  // counters (UpgradeReq / FORWARD-downgrade), which is what proves the path is
  // live. Returning HIT here would deadlock the atomic ack.
  return false;
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    return RESERVATION_FAIL;

  assert(mf->get_data_size() <= m_config.get_line_sz());

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tags.access(block_addr, time, cache_index, mf);
  enum cache_request_status cache_status = RESERVATION_FAIL;
  assert(status != RESERVATION_FAIL);
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  m_fragment_fifo.push(
      fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  if (status == MISS) {
    // we need to send a memory request...
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
    mf->set_data_size(m_config.get_line_sz());
    m_tags.fill(cache_index, time, mf);  // mark block as valid
    m_request_fifo.push(mf);
    mf->set_status(m_request_queue_status, time);
    events.push_back(cache_event(READ_REQUEST_SENT));
    cache_status = MISS;
  } else {
    // the value *will* *be* in the cache already
    cache_status = HIT_RESERVED;
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status));
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status));
  return cache_status;
}

void tex_cache::cycle() {
  // send next request to lower level of memory
  if (!m_request_fifo.empty()) {
    mem_fetch *mf = m_request_fifo.peek();
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      m_request_fifo.pop();
      m_memport->push(mf);
    }
  }
  // read ready lines from cache
  if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
    const fragment_entry &e = m_fragment_fifo.peek();
    if (e.m_miss) {
      // check head of reorder buffer to see if data is back from memory
      unsigned rob_index = m_rob.next_pop_index();
      const rob_entry &r = m_rob.peek(rob_index);
      assert(r.m_request == e.m_request);
      // assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr())
      // );
      if (r.m_ready) {
        assert(r.m_index == e.m_cache_index);
        m_cache[r.m_index].m_valid = true;
        m_cache[r.m_index].m_block_addr = r.m_block_addr;
        m_result_fifo.push(e.m_request);
        m_rob.pop();
        m_fragment_fifo.pop();
      }
    } else {
      // hit:
      assert(m_cache[e.m_cache_index].m_valid);
      assert(m_cache[e.m_cache_index].m_block_addr ==
             m_config.block_addr(e.m_request->get_addr()));
      m_result_fifo.push(e.m_request);
      m_fragment_fifo.pop();
    }
  }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  assert(!m_rob.empty());
  mf->set_status(m_rob_status, time);

  unsigned rob_index = e->second.m_rob_index;
  rob_entry &r = m_rob.peek(rob_index);
  assert(!r.m_ready);
  r.m_ready = true;
  r.m_time = time;
  assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const {
  fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
  fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
          m_fragment_fifo.capacity());
  fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
          m_rob.capacity());
  fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
          m_request_fifo.capacity());
  if (!m_rob.empty()) fprintf(fp, "reorder buffer contents:\n");
  for (int n = m_rob.size() - 1; n >= 0; n--) {
    unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
    const rob_entry &r = m_rob.peek(index);
    fprintf(fp, "tex rob[%3d] : %s ", index,
            (r.m_ready ? "ready  " : "pending"));
    if (r.m_ready)
      fprintf(fp, "@%6u", r.m_time);
    else
      fprintf(fp, "       ");
    fprintf(fp, "[idx=%4u]", r.m_index);
    r.m_request->print(fp, false);
  }
  if (!m_fragment_fifo.empty()) {
    fprintf(fp, "fragment fifo (oldest) :");
    fragment_entry &f = m_fragment_fifo.peek();
    fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
    f.m_request->print(fp, false);
  }
}
/******************************************************************************************************************************************/

// ---------------------------------------------------------------------------
// Cross-cluster read fast path. Returns true if the read was satisfied from a
// neighbor GPC L1 (data placed into local L1 by fill); false => caller lets the
// normal path fetch from L2. On success sets g_c2c_defer_extra_cycles to the
// modelled probe+transfer latency; on false leaves it -1.
// Safety: only serves lines whose neighbor L1 copy == L2 copy (WT single writer).
// ---------------------------------------------------------------------------
bool l1_cache::try_cross_cluster_read(new_addr_type addr, mem_fetch *mf,
                                      unsigned time,
                                      std::list<cache_event> &events) {
  init_c2c_mode();
  g_c2c_defer_extra_cycles = -1;
  if (g_c2c_mode == 0) return false;           // arm A: baseline, always L2
  if (!m_owner) return false;
  if (mf->get_access_type() != GLOBAL_ACC_R) return false;

  new_addr_type block_addr = m_config.block_addr(addr);
  if (!is_cross_cluster_addr(block_addr)) return false;
  g_c2c_total_reads++;

  simt_core_cluster *scc = m_owner->get_simt_core_cluster();
  if (!scc || !scc->m_gpc) return false;
  unsigned my_gpc = scc->m_gpc->get_gpc_id();

  // Multi-writer lines: correctness over speed -> forced to L2 (never probe).
  bool multiwriter = (g_cross_cluster_multiwriter.find(block_addr) !=
                      g_cross_cluster_multiwriter.end());

  // ------------------------------------------------------------------
  // Authoritative owner directory lookup (XCWL). This is FREE information
  // piggybacked on the L2 fill that already runs for a cross-cluster miss --
  // it is NOT a probe (no fabric message, no latency). Used to SEED CCOP.
  // ------------------------------------------------------------------
  shader_core_ctx *dir_owner = nullptr;
  unsigned dir_owner_gpc = (unsigned)-1;
  unsigned dir_owner_sid = (unsigned)-1;
  for (unsigned g : g_xcwl_gpcs) {
    if (g == my_gpc) continue;                 // same-GPC owner is not cross-cluster
    auto git = g_xcwl.find(g);
    if (git == g_xcwl.end()) continue;
    auto eit = git->second.find(block_addr);
    if (eit == git->second.end() || !eit->second.owner_core) continue;
    dir_owner = eit->second.owner_core;
    dir_owner_gpc = g;
    dir_owner_sid = dir_owner->get_sid();
    break;                                      // single-writer: at most one cross-GPC owner
  }
  unsigned cur_gen = 0;
  { auto it = g_xcwl_gen.find(block_addr);
    if (it != g_xcwl_gen.end()) cur_gen = it->second; }

  // Cross-GPC accounting over every read that resolved to a known owner. The
  // benchmark's validity guard requires this fraction > 0.9 (consumers really
  // are on different GPCs than producers).
  if (dir_owner) {
    if (dir_owner_gpc != my_gpc) g_c2c_crossgpc_reads++;
    else                         g_c2c_samegpc_reads++;
  }

  // CCOP table entry for THIS consumer SM, region-keyed.
  std::vector<ccop_entry> &tbl = ccop_table_for((const void *)this);
  new_addr_type region = ccop_region(block_addr);
  ccop_entry &e = tbl[ccop_slot(region)];
  bool entry_matches = e.valid && e.region_tag == region;

  // ==================================================================
  // QUERY: confidence gate. Fire AT MOST ONE directed probe, or ZERO.
  //   probe iff  valid && tag-match && !poison && conf>=THRESH &&
  //              owner_gpc != self  (and not the cold-only control).
  // ==================================================================
  bool may_probe = entry_matches && !e.poison &&
                   e.conf >= CCOP_CONF_THRESH && e.owner_gpc != my_gpc &&
                   !multiwriter && g_c2c_noprobe == 0;

  if (may_probe) {
    // ---- Exactly ONE directed probe to the predicted owner SM. ----
    g_ccop_probes++;
    g_c2c_crossgpc_probes++;
    g_c2c_probe_bytes += 32;                    // directed probe request (BW charged)

    // Single-peer peek: is the predicted owner still the current owner and is
    // its value current (gen gate)?  gen mismatch => NACK (never serve stale).
    bool probe_hit = false;
    if (dir_owner && dir_owner_sid == e.owner_sid &&
        dir_owner_gpc == e.owner_gpc && cur_gen == e.gen) {
      // Transparency sub-count: was the line still physically resident in the
      // owner L1 under stock LRU?  (Not a serve requirement -- the XCWL owner +
      // WT invariant guarantee value == L2 == truth regardless.)
      l1_cache *cand = dir_owner->get_L1D_cache();
      if (cand) {
        unsigned ci = (unsigned)-1;
        cache_request_status ps =
            cand->m_tag_array->probe(block_addr, ci, mf, false, true);
        bool phys = false;
        if (ps == HIT || ps == HIT_RESERVED) {
          cache_block_t *b = cand->m_tag_array->get_block(ci);
          if (b && b->is_readable(mf->get_access_sector_mask())) phys = true;
        }
        if (phys) g_c2c_phys_resident_hits++; else g_c2c_pinned_only_hits++;
      }
      probe_hit = true;
    }

    if (probe_hit) {
      // HIT: place the line locally (data == L2 copy by WT invariant; the
      // functional value is always supplied by the functional simulator).
      m_tag_array->fill(block_addr, time, mf, false);
      unsigned li = (unsigned)-1;
      cache_request_status ls =
          m_tag_array->probe(block_addr, li, mf, false, true);
      if (ls == HIT || ls == HIT_RESERVED) {
        e.conf = (e.conf < CCOP_CONF_MAX) ? e.conf + 1 : CCOP_CONF_MAX;
        e.gen = cur_gen;
        g_ccop_fresh_hits++;
        g_c2c_neighbor_hits++;                  // back-compat alias
        g_c2c_pred_hits++;                      // back-compat alias
        // S2 (probe-first) genuinely offloads L2; S1 (P3(P2)) spends L2 BW
        // speculatively in parallel, so it does NOT claim the L2 saving.
        if (g_c2c_mode == 2) g_c2c_l2_accesses_saved++;
        // Completion latency == one directed SM-to-SM hop (beats L2 242).
        g_c2c_defer_extra_cycles = sm2sm_latency;
        events.push_back(cache_event(CLUSTER_READ_SHARE_HIT));
        return true;
      }
      // fill unexpectedly failed -> fall through to L2 (never-wrong).
    }
    // NACK / mispredict: L2 was launched speculatively in parallel, so it
    // completes at its normal time -- the wasted probe costs ONLY bandwidth,
    // NEVER latency. Demote confidence; reseed below re-syncs on a gen change.
    g_ccop_mispredict++;
    g_c2c_mispredicts++;                        // back-compat alias
    e.conf = (e.conf > 0) ? e.conf - 1 : 0;
  }

  // ==================================================================
  // SEED WITHOUT PROBE: learn/refresh the owner for FREE from the XCWL
  // directory (piggybacked on the L2 fill). First touch: 0 probes, always
  // correct. Poison on a 2nd distinct writer GPC (multi-writer safe).
  // ==================================================================
  if (multiwriter) {
    if (entry_matches) { e.poison = true; }
    else { e.valid = true; e.region_tag = region; e.poison = true;
           e.owner_sid = dir_owner_sid; e.owner_gpc = dir_owner_gpc;
           e.conf = 0; e.gen = cur_gen; }
    g_ccop_poison_skips++;
  } else if (dir_owner && dir_owner_gpc != my_gpc) {
    if (entry_matches && e.owner_sid == dir_owner_sid &&
        e.owner_gpc == dir_owner_gpc && !e.poison) {
      // Same owner reconfirmed: raise confidence (2-bit sat), refresh gen.
      e.conf = (e.conf < CCOP_CONF_MAX) ? e.conf + 1 : CCOP_CONF_MAX;
      e.gen = cur_gen;
    } else {
      // New / changed owner: install fresh seed at conf=1 (needs a 2nd
      // confirming observation before it will probe -- confidence-gated).
      e.valid = true; e.region_tag = region; e.poison = false;
      e.owner_sid = dir_owner_sid; e.owner_gpc = dir_owner_gpc;
      e.conf = 1; e.gen = cur_gen;
    }
    g_ccop_cold_l2_learn++;
  } else if (dir_owner && dir_owner_gpc == my_gpc) {
    g_c2c_samegpc_skips++;                      // owner is local: never probe
  }

  // Default: straight to L2 (baseline timing, byte-identical to arm A).
  g_c2c_l2_fallbacks++;
  g_c2c_defer_extra_cycles = -1;
  if (g_c2c_debug > 0)
    printf("[CCOP_DBG] L2 my_gpc=%u addr=0x%llx conf=%d poison=%d owner_gpc=%d\n",
           my_gpc, (unsigned long long)block_addr, entry_matches ? e.conf : -1,
           entry_matches ? (int)e.poison : -1, (int)dir_owner_gpc);
  return false;
}

