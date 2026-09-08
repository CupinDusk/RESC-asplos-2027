import re, sys, io

# ---------- 1. abstract_hardware_model.h : add the config field ----------
p = "src/abstract_hardware_model.h"
s = open(p).read()
anchor = """  uint32_t cluster_arrive_latency;
  uint32_t cluster_wait_latency;
};"""
assert s.count(anchor) == 1, "anchor1 count=%d" % s.count(anchor)
new = """  uint32_t cluster_arrive_latency;
  uint32_t cluster_wait_latency;
  // FIDELITY FIX D2 (2026-09-03): model the Hopper cluster-barrier ACQUIRE.
  // Real sm_90 lowers barrier.cluster.wait{.acquire} (acquire is the PTX
  // default, PTX ISA 9.3 s9.7.14.3) to UCGABAR_WAIT; CCTL.IVALL -- a whole-L1
  // invalidate.  1 = model it (faithful to shipping silicon).
  // 0 = model fence.acquire.sync_restrict::shared::cluster.cluster
  //     (PTX ISA 9.3 s9.7.14.4), whose effect is confined to .shared::cluster
  //     state space and therefore leaves the global L1 intact -- the legal
  //     escape a SMEM-staging baseline may take.
  bool cluster_barrier_invalidate_l1;
};"""
s = s.replace(anchor, new)
open(p, "w").write(s)
print("OK abstract_hardware_model.h")

# ---------- 2. gpu-sim.cc : register the option ----------
p = "src/gpgpu-sim/gpu-sim.cc"
s = open(p).read()
anchor = """  option_parser_register(opp, "-gpgpu_cluster_wait_latency", OPT_INT32,
                         &cluster_wait_latency,
                         "Minimum latency for cluster.wait", "50");"""
assert s.count(anchor) == 1, "anchor2 count=%d" % s.count(anchor)
new = anchor + """
  // FIDELITY FIX D2 (2026-09-03): see abstract_hardware_model.h.
  option_parser_register(opp, "-gpgpu_cluster_barrier_invalidate_l1", OPT_BOOL,
                         &cluster_barrier_invalidate_l1,
                         "Invalidate L1 at a cluster-barrier acquire, as real "
                         "sm_90 does via CCTL.IVALL (1); 0 models the "
                         "fence.acquire.sync_restrict::shared::cluster.cluster "
                         "escape, which leaves the global L1 intact", "1");"""
s = s.replace(anchor, new)
open(p, "w").write(s)
print("OK gpu-sim.cc")

# ---------- 3. shader.cc : perform the invalidate on the completing acquire ----
p = "src/gpgpu-sim/shader.cc"
s = open(p).read()
anchor = """      case WAIT: {
        assert(m_shader->m_warp[warp_id]->m_sync_latency == 0);
        m_shader->m_warp[warp_id]->m_sync_latency =
            m_shader->get_config()->cluster_wait_latency;
        m_ptx_cluster_info.at(warp_id) =
            m_shader->get_thread_info().at(warp_id * 32)->m_cluster_info;
        break;
      }"""
assert s.count(anchor) == 1, "anchor3 count=%d" % s.count(anchor)
new = """      case WAIT: {
        assert(m_shader->m_warp[warp_id]->m_sync_latency == 0);
        m_shader->m_warp[warp_id]->m_sync_latency =
            m_shader->get_config()->cluster_wait_latency;
        m_ptx_cluster_info.at(warp_id) =
            m_shader->get_thread_info().at(warp_id * 32)->m_cluster_info;
        // ---- FIDELITY FIX D2 (2026-09-03) ------------------------------
        // barrier.cluster.wait carries an implicit .acquire (PTX ISA 9.3
        // s9.7.14.3: "If the optional .acquire qualifier is absent for
        // barrier.cluster.wait, .acquire is assumed by default"), and on
        // shipping sm_90 that acquire lowers to UCGABAR_WAIT; CCTL.IVALL --
        // a whole-L1 invalidate (SASS-verified on real H20).  ClusterSim
        // previously modelled the cluster barrier as latency + rendezvous
        // only, so no simulated arm ever paid this cost.  Charge it here,
        // reusing the identical whole-tag-array invalidate that membar.gl
        // already uses (shader.cc:4069-4080 -> gpu-cache.cc:934-945).
        //
        // Gated OFF to model fence.acquire.sync_restrict::shared::cluster
        // .cluster (PTX ISA 9.3 s9.7.14.4), which restricts the acquire to
        // .shared::cluster state space and provably costs the global L1
        // nothing (measured: single NOP on sm_90a).  Both forms are legal
        // ISA, so both must be modelled; charging the escape would charge
        // software a cost the ISA lets it avoid.
        if (m_shader->get_config()->cluster_barrier_invalidate_l1) {
          m_shader->cluster_acquire_invalidate_l1();
        }
        // ---------------------------------------------------------------
        break;
      }"""
s = s.replace(anchor, new)

# add the shader_core_ctx helper next to cache_invalidate()
anchor2 = """void shader_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }"""
assert s.count(anchor2) == 1, "anchor4 count=%d" % s.count(anchor2)
new2 = anchor2 + """

// FIDELITY FIX D2 (2026-09-03): whole-L1 invalidate charged at a cluster-barrier
// acquire, modelling sm_90's CCTL.IVALL.  Separate entry point from
// cache_invalidate() purely so the event can be counted independently.
unsigned long long g_cluster_acquire_invalidates = 0;
void shader_core_ctx::cluster_acquire_invalidate_l1() {
  g_cluster_acquire_invalidates++;
  m_ldst_unit->invalidate();
}"""
s = s.replace(anchor2, new2)
open(p, "w").write(s)
print("OK shader.cc")

# ---------- 4. shader.h : declare the helper ----------
p = "src/gpgpu-sim/shader.h"
s = open(p).read()
anchor = """  void cache_flush();
  void cache_invalidate();
  void accept_fetch_response(mem_fetch *mf);"""
assert s.count(anchor) == 1, "anchor5 count=%d" % s.count(anchor)
new = """  void cache_flush();
  void cache_invalidate();
  // FIDELITY FIX D2 (2026-09-03): cluster-barrier acquire invalidate (CCTL.IVALL)
  void cluster_acquire_invalidate_l1();
  void accept_fetch_response(mem_fetch *mf);"""
s = s.replace(anchor, new)
open(p, "w").write(s)
print("OK shader.h")
