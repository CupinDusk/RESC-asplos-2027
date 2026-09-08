#!/usr/bin/env python3
# Historical execution guard; no server is configured here.
raise SystemExit('Historical RESC recipe: execution disabled after infrastructure sanitation. Review a new laboratory-only launcher before running.')
# SELECTIVE-INVALIDATE patch (2026-09-03).
# Pre-registration: resc-asplos-2027/selective-invalidate-prereg.md
#   SHA-256 8ebabe1994424f2739d13fa47b7c2bdefc3b9171a625ca1e07dab740a704e618
#   frozen 2026-09-03T11:11:43Z (BEFORE this file was written).
#
# Extends -gpgpu_cluster_barrier_invalidate_l1 from a bool to a 4-valued knob:
#   0 = no invalidate                (fence.acquire.sync_restrict escape)   [UNCHANGED]
#   1 = whole-L1 invalidate          (CCTL.IVALL, shipping silicon)         [UNCHANGED]
#   2 = marked-line selective        (RESC minimal-correct acquire)         [NEW]
#   3 = marked-line selective, owner-retaining (RESC + DeNovo/hLRC SI rule) [NEW]
#
# Values 0 and 1 must remain BIT-IDENTICAL to the pre-change build.
import sys, os

os.chdir("/legacy-research/owner/cudatest/RESC-c2c-wt")
def rd(p): return open(p).read()
def wr(p, s): open(p, "w").write(s)
def sub(p, old, new, tag):
    s = rd(p)
    n = s.count(old)
    assert n == 1, "%s: anchor count=%d" % (tag, n)
    wr(p, s.replace(old, new))
    print("OK", tag)

# ===========================================================================
# 1. abstract_hardware_model.h -- widen the field bool -> int
# ===========================================================================
sub("src/abstract_hardware_model.h",
"""  //     state space and therefore leaves the global L1 intact -- the legal
  //     escape a SMEM-staging baseline may take.
  bool cluster_barrier_invalidate_l1;""",
"""  //     state space and therefore leaves the global L1 intact -- the legal
  //     escape a SMEM-staging baseline may take.
  // SELECTIVE-INVALIDATE (2026-09-03): widened bool -> int.  RESC's defining
  // act is neither 0 nor 1 but a THIRD behaviour -- invalidate only the lines
  // that could be stale (the marked communication set) and preserve the rest:
  //   2 = marked-line selective invalidate (RESC minimal-correct acquire)
  //   3 = 2, plus retain marked lines held in an OWNER cluster-state
  //       (the DeNovo/hLRC self-invalidation rule: an acquire self-invalidates
  //       valid-but-not-owned lines; an owner's copy is the newest).
  // Values 2/3 model RESC HARDWARE and must never be granted to a baseline
  // arm that has no such hardware (fairness rule, prereg S3).
  int cluster_barrier_invalidate_l1;""",
"ahm.h field")

# ===========================================================================
# 2. gpu-sim.cc -- OPT_BOOL -> OPT_INT32
# ===========================================================================
sub("src/gpgpu-sim/gpu-sim.cc",
"""  option_parser_register(opp, "-gpgpu_cluster_barrier_invalidate_l1", OPT_BOOL,
                         &cluster_barrier_invalidate_l1,
                         "Invalidate L1 at a cluster-barrier acquire, as real "
                         "sm_90 does via CCTL.IVALL (1); 0 models the "
                         "fence.acquire.sync_restrict::shared::cluster.cluster "
                         "escape, which leaves the global L1 intact", "1");""",
"""  option_parser_register(opp, "-gpgpu_cluster_barrier_invalidate_l1", OPT_INT32,
                         &cluster_barrier_invalidate_l1,
                         "Cluster-barrier acquire behaviour: 0 = no L1 "
                         "invalidate (models fence.acquire.sync_restrict::"
                         "shared::cluster.cluster); 1 = whole-L1 invalidate as "
                         "real sm_90 does via CCTL.IVALL; 2 = RESC marked-line "
                         "SELECTIVE invalidate (unmarked lines preserved); "
                         "3 = 2 but marked lines in an owner cluster-state are "
                         "also retained (DeNovo/hLRC self-invalidation rule)",
                         "1");""",
"gpu-sim.cc option")

# ===========================================================================
# 3. gpu-cache.cc -- the selective invalidate itself, on the marked registry
# ===========================================================================
sub("src/gpgpu-sim/gpu-cache.cc",
"""void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

  m_dirty = 0;
  is_used = false;
}""",
"""void tag_array::invalidate() {
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
}""",
"gpu-cache.cc selective_invalidate")

# ===========================================================================
# 4. gpu-cache.h -- declare tag_array::selective_invalidate + l1_cache wrapper
# ===========================================================================
sub("src/gpgpu-sim/gpu-cache.h",
"""  void flush();       // flush all written entries
  void invalidate();  // invalidate all entries
  void new_window();""",
"""  void flush();       // flush all written entries
  void invalidate();  // invalidate all entries
  // SELECTIVE-INVALIDATE (2026-09-03): invalidate ONLY lines in the RESC
  // marked-line registry, preserving unmarked lines.  owner_retaining also
  // preserves marked lines held in an owner cluster-state (DeNovo/hLRC
  // self-invalidation rule).  See gpu-cache.cc.
  void selective_invalidate(bool have_key, unsigned long long key,
                            bool owner_retaining);
  void new_window();""",
"gpu-cache.h tag_array decl")

sub("src/gpgpu-sim/gpu-cache.h",
"""  void flush() { m_tag_array->flush(); }
  void invalidate() { m_tag_array->invalidate(); }""",
"""  void flush() { m_tag_array->flush(); }
  void invalidate() { m_tag_array->invalidate(); }
  // SELECTIVE-INVALIDATE (2026-09-03)
  void selective_invalidate(bool have_key, unsigned long long key,
                            bool owner_retaining) {
    m_tag_array->selective_invalidate(have_key, key, owner_retaining);
  }""",
"gpu-cache.h baseline_cache wrapper")

# ===========================================================================
# 5. shader.cc / shader.h -- ldst_unit + shader_core_ctx plumbing
# ===========================================================================
sub("src/gpgpu-sim/shader.cc",
"""void ldst_unit::invalidate() {
  // Flush L1D cache
  m_L1D->invalidate();
}""",
"""void ldst_unit::invalidate() {
  // Flush L1D cache
  m_L1D->invalidate();
}

// SELECTIVE-INVALIDATE (2026-09-03): marked-line-only L1 invalidate.
void ldst_unit::selective_invalidate(bool have_key, unsigned long long key,
                                     bool owner_retaining) {
  if (m_L1D) m_L1D->selective_invalidate(have_key, key, owner_retaining);
}""",
"shader.cc ldst_unit::selective_invalidate")

sub("src/gpgpu-sim/shader.h",
"""  void flush();
  void invalidate();""",
"""  void flush();
  void invalidate();
  // SELECTIVE-INVALIDATE (2026-09-03)
  void selective_invalidate(bool have_key, unsigned long long key,
                            bool owner_retaining);""",
"shader.h ldst_unit decl")

sub("src/gpgpu-sim/shader.cc",
"""unsigned long long g_cluster_acquire_invalidates = 0;
void shader_core_ctx::cluster_acquire_invalidate_l1() {
  g_cluster_acquire_invalidates++;
  m_ldst_unit->invalidate();
}""",
"""unsigned long long g_cluster_acquire_invalidates = 0;
void shader_core_ctx::cluster_acquire_invalidate_l1() {
  g_cluster_acquire_invalidates++;
  m_ldst_unit->invalidate();
}

// SELECTIVE-INVALIDATE (2026-09-03): the RESC form of the cluster-acquire.
// mode 2 = marked-line selective; mode 3 = selective + owner-retaining.
// The (gpc,slot) key identifies WHICH cluster's marked set applies -- the same
// key the read-share path uses (gpu-cache.cc resc_cluster_key_for_mf).
unsigned long long g_cluster_acquire_selective_invalidates = 0;
void shader_core_ctx::cluster_acquire_selective_invalidate_l1(
    unsigned warp_id, int mode) {
  g_cluster_acquire_selective_invalidates++;
  bool have_key = false;
  unsigned long long key = 0;
  simt_core_cluster *scc = get_simt_core_cluster();
  if (scc && scc->m_gpc) {
    unsigned gpc_id = scc->m_gpc->get_gpc_id();
    unsigned slot = get_warp_cluster_slot(warp_id);
    key = (static_cast<unsigned long long>(gpc_id) << 32) |
          static_cast<unsigned long long>(slot);
    have_key = true;
  }
  m_ldst_unit->selective_invalidate(have_key, key, /*owner_retaining=*/mode == 3);
}""",
"shader.cc cluster_acquire_selective_invalidate_l1")

sub("src/gpgpu-sim/shader.h",
"""  // FIDELITY FIX D2 (2026-09-03): cluster-barrier acquire invalidate (CCTL.IVALL)
  void cluster_acquire_invalidate_l1();""",
"""  // FIDELITY FIX D2 (2026-09-03): cluster-barrier acquire invalidate (CCTL.IVALL)
  void cluster_acquire_invalidate_l1();
  // SELECTIVE-INVALIDATE (2026-09-03): RESC marked-line-only acquire (mode 2/3)
  void cluster_acquire_selective_invalidate_l1(unsigned warp_id, int mode);""",
"shader.h shader_core_ctx decl")

# ---- the barrier WAIT dispatch: 0 none / 1 whole / 2,3 selective ----------
sub("src/gpgpu-sim/shader.cc",
"""        if (m_shader->get_config()->cluster_barrier_invalidate_l1) {
          m_shader->cluster_acquire_invalidate_l1();
        }
        // ---------------------------------------------------------------""",
"""        // SELECTIVE-INVALIDATE (2026-09-03): 0 = none (sync_restrict escape),
        // 1 = whole-L1 (CCTL.IVALL, silicon), 2 = RESC marked-line selective,
        // 3 = selective + owner-retaining.  2/3 model RESC HARDWARE and are
        // barred from baseline arms by the experiment's fairness rule; the
        // simulator itself does not police that (a config knob cannot know
        // which arm is running) -- it is enforced by the runner and recorded
        // in the pre-registration.
        {
          const int __inv_mode =
              m_shader->get_config()->cluster_barrier_invalidate_l1;
          if (__inv_mode == 1) {
            m_shader->cluster_acquire_invalidate_l1();
          } else if (__inv_mode == 2 || __inv_mode == 3) {
            m_shader->cluster_acquire_selective_invalidate_l1(warp_id,
                                                              __inv_mode);
          }
        }
        // ---------------------------------------------------------------""",
"shader.cc barrier WAIT dispatch")

# ===========================================================================
# 6. gpu-cache.cc -- marked/unmarked SPLIT L1 read hit/miss counters.
#    This is the KILL-S4 instrument: a cycle difference without the
#    "unmarked hit rate preserved under 2, destroyed under 1" signature is
#    not the claimed mechanism.  Counted at the single L1 entry point that
#    every global read passes through, AFTER the probe and BEFORE any RESC
#    fast path can rewrite the status.
# ===========================================================================
sub("src/gpgpu-sim/gpu-cache.cc",
"""  // RESC cross-cluster (inter-GPC) fast path (NEW). Self-contained;""",
"""  // SELECTIVE-INVALIDATE (2026-09-03) KILL-S4 instrument: split the L1 read
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

  // RESC cross-cluster (inter-GPC) fast path (NEW). Self-contained;""",
"gpu-cache.cc split hit/miss counters")

sub("src/gpgpu-sim/gpu-cache.cc",
"""unsigned long long g_selinv_events            = 0;  // selective invalidate calls""",
"""unsigned long long g_l1_rd_marked_hit    = 0;
unsigned long long g_l1_rd_marked_miss   = 0;
unsigned long long g_l1_rd_unmarked_hit  = 0;
unsigned long long g_l1_rd_unmarked_miss = 0;
unsigned long long g_selinv_events            = 0;  // selective invalidate calls""",
"gpu-cache.cc split counter defs")

# ===========================================================================
# 7. gpu-sim.cc -- print the new counters next to [RESC_GEN_STATS]
# ===========================================================================
sub("src/gpgpu-sim/gpu-sim.cc",
"""    printf("[RESC_GEN_STATS] gen_ring=%d epoch_ticks=%llu readshare_served=%llu "
           "readshare_expired=%llu\\n",
           g_resc_gen_ring, g_resc_epoch_ticks, g_resc_gen_served,
           g_resc_gen_expired);
  }""",
"""    printf("[RESC_GEN_STATS] gen_ring=%d epoch_ticks=%llu readshare_served=%llu "
           "readshare_expired=%llu\\n",
           g_resc_gen_ring, g_resc_epoch_ticks, g_resc_gen_served,
           g_resc_gen_expired);
  }
  {
    // SELECTIVE-INVALIDATE (2026-09-03): the selective-acquire counters and the
    // marked/unmarked split L1 read hit/miss counts (KILL-S4 instrument).
    extern unsigned long long g_cluster_acquire_invalidates,
        g_cluster_acquire_selective_invalidates;
    extern unsigned long long g_selinv_events, g_selinv_lines_examined,
        g_selinv_marked_invalidated, g_selinv_unmarked_preserved,
        g_selinv_owner_retained;
    extern unsigned long long g_l1_rd_marked_hit, g_l1_rd_marked_miss,
        g_l1_rd_unmarked_hit, g_l1_rd_unmarked_miss;
    printf("[RESC_SELINV_STATS] whole_invalidates=%llu selective_invalidates=%llu "
           "selinv_events=%llu lines_examined=%llu marked_invalidated=%llu "
           "unmarked_preserved=%llu owner_retained=%llu\\n",
           g_cluster_acquire_invalidates, g_cluster_acquire_selective_invalidates,
           g_selinv_events, g_selinv_lines_examined, g_selinv_marked_invalidated,
           g_selinv_unmarked_preserved, g_selinv_owner_retained);
    unsigned long long mtot = g_l1_rd_marked_hit + g_l1_rd_marked_miss;
    unsigned long long utot = g_l1_rd_unmarked_hit + g_l1_rd_unmarked_miss;
    printf("[RESC_L1SPLIT_STATS] marked_rd_hit=%llu marked_rd_miss=%llu "
           "marked_hitrate=%.6f unmarked_rd_hit=%llu unmarked_rd_miss=%llu "
           "unmarked_hitrate=%.6f\\n",
           g_l1_rd_marked_hit, g_l1_rd_marked_miss,
           mtot ? (double)g_l1_rd_marked_hit / (double)mtot : 0.0,
           g_l1_rd_unmarked_hit, g_l1_rd_unmarked_miss,
           utot ? (double)g_l1_rd_unmarked_hit / (double)utot : 0.0);
  }""",
"gpu-sim.cc selinv stats print")

# ===========================================================================
# 8. RESC_QUIET -- suppress the multi-MB per-event debug printf that caused
#    timeouts in the previous experiment.  Default = current (loud) behaviour,
#    so it changes nothing unless explicitly set.  DISCLOSED in the methodology.
# ===========================================================================
sub("src/gpgpu-sim/gpu-cache.cc",
"""  const unsigned long long key = make_cluster_key(gpc_id, cluster_slot);
  g_cluster_coherent_addrs_by_cluster[key].insert(addr);
  printf("\\n[RESC] Registered cluster coherent address: 0x%llx (gpc=%u slot=%u)\\n",
         (unsigned long long)addr, gpc_id, cluster_slot);""",
"""  const unsigned long long key = make_cluster_key(gpc_id, cluster_slot);
  g_cluster_coherent_addrs_by_cluster[key].insert(addr);
  // SELECTIVE-INVALIDATE (2026-09-03): RESC_QUIET=1 suppresses this per-event
  // printf (multi-MB per run; caused timeouts in the previous experiment).
  // Default unset == current loud behaviour, so nothing changes unless asked.
  if (!resc_quiet())
  printf("\\n[RESC] Registered cluster coherent address: 0x%llx (gpc=%u slot=%u)\\n",
         (unsigned long long)addr, gpc_id, cluster_slot);""",
"gpu-cache.cc quiet registration printf")

sub("src/gpgpu-sim/gpu-cache.cc",
"""static inline unsigned long long make_cluster_key(unsigned gpc_id,
                                                  unsigned cluster_slot) {
  return (static_cast<unsigned long long>(gpc_id) << 32) |
         static_cast<unsigned long long>(cluster_slot);
}""",
"""static inline unsigned long long make_cluster_key(unsigned gpc_id,
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
}""",
"gpu-cache.cc resc_quiet helper")

sub("src/gpgpu-sim/gpu-cache.cc",
"""        printf("\\n[RESC] Read share SUCCESS (special) - addr=0x%llx, owner_sid=%u\\n", addr, m_owner->get_sid());""",
"""        if (!resc_quiet())
        printf("\\n[RESC] Read share SUCCESS (special) - addr=0x%llx, owner_sid=%u\\n", addr, m_owner->get_sid());""",
"gpu-cache.cc quiet readshare special")

sub("src/gpgpu-sim/gpu-cache.cc",
"""        printf("\\n[RESC] Read share SUCCESS (normal) - addr=0x%llx, owner_sid=%u\\n", addr, m_owner->get_sid());""",
"""        if (!resc_quiet())
        printf("\\n[RESC] Read share SUCCESS (normal) - addr=0x%llx, owner_sid=%u\\n", addr, m_owner->get_sid());""",
"gpu-cache.cc quiet readshare normal")

# instructions.cc: the interceptor emits 15 [CLUSTER_STATE] printf per registered
# line (millions of lines of output on a marked-heavy run).  Silence them all under
# RESC_QUIET by routing them through a helper that is a no-op when quiet.  Purely
# an output change; no control flow, no counter, no timing is touched.
s = rd("src/cuda-sim/instructions.cc")
n_before = s.count('printf("\\n[CLUSTER_STATE]')
assert n_before == 15, "instructions.cc CLUSTER_STATE printf count=%d" % n_before
s = s.replace('printf("\\n[CLUSTER_STATE]', 'resc_qprintf("\\n[CLUSTER_STATE]')
anchor = """extern "C" void gpgpusim_register_cluster_coherent_addr(new_addr_type addr,"""
assert s.count(anchor) == 1
s = s.replace(anchor,
"""// SELECTIVE-INVALIDATE (2026-09-03): RESC_QUIET=1 silences the per-registration
// [CLUSTER_STATE] trace (15 printf per marked line; multi-MB per run, and the
// direct cause of the timeouts in the previous experiment).  Output only.
static int g_resc_q_ins = -1;
static void resc_qprintf(const char *fmt, ...) {
  if (g_resc_q_ins < 0) {
    const char *q = getenv("RESC_QUIET");
    g_resc_q_ins = (q && atoi(q)) ? 1 : 0;
  }
  if (g_resc_q_ins) return;
  va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
}

""" + anchor)
wr("src/cuda-sim/instructions.cc", s)
print("OK instructions.cc quiet (15 printf routed through resc_qprintf)")

print("ALL PATCHES APPLIED")
