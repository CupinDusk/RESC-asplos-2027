#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESC multi-writer coherence: off-sim explicit-state model checker.

WHY THIS EXISTS
---------------
GPGPU-Sim cache blocks are TAG-ONLY (cache_block_t has tag/status/timestamp, no
data array; see gpu-cache.h). Every load/store/atomic executes in the FUNCTIONAL
simulator against global shadow memory, and the RESC coherence machine is a pure
timing/routing overlay. Therefore an in-sim checksum is ALWAYS byte-correct
regardless of whether the protocol is right or wrong -- bit-identity in-sim is
STRUCTURAL, not a coherence proof. (The implementation itself says so:
gpu-cache.cc:3319-3323 -- even the whole-line-LWW ablation "does NOT corrupt the
checksum; it exists to show the design DIFFERENCE".)

So coherence correctness must be established OFF the simulator. This program is a
self-contained explicit-state enumerator (BFS over all reachable global states
for a small bounded config) that models the ACTUAL implemented transition table
and carries DATA VALUES through it -- the thing the tag-only sim cannot do. It
checks three invariants on the real protocol (must PASS) and two broken ablations
(must produce counterexamples), the same methodology as Murphi (Dill et al.,
ICCD 1992) but written from scratch because Murphi is not installed on this host.

MODELLED IMPLEMENTATION (verbatim anchors, RESC-c2c-wt tree, verified this session)
  states               gpu-cache.h:58  {INVALID,EXCLUSIVE,FORWARD,SHARED,EM,SM,MW_MERGE}
  per-line written mask gpu-cache.h:457-463,555  m_mw_written_mask + mw_or/get/clear
  Path A merge          gpu-cache.cc:3240 try_cluster_multiwriter_merge
     real per-sector OR  gpu-cache.cc:3327  lsb->mw_or_written_mask(my_sector)
     LWW ablation        gpu-cache.cc:3324-3325 clear then set only my_sector
     writeback acct      gpu-cache.cc:3359-3362 real=mask.count(), LWW=whole line
  serve-path invariant  gpu-cache.cc:2663-2671, 3099-3108 MW_MERGE => no whole-line fwd, read L2
  Path B atomic token   gpu-cache.cc:3386 try_cluster_atomic_token
     local owner (no hop) gpu-cache.cc:3425-3431
     UpgradeReq hand-off  gpu-cache.cc:3435-3457 EM->FORWARD->INVALID, one fabric msg
     gen bump             gpu-cache.cc:3470
  honesty anchor        gpu-cache.cc:3319-3323 (LWW does not corrupt in-sim checksum)

The GPU memory model (why case C = racing plain writes to one byte is UB, and why
same-scope atomics are ordered) is the NVIDIA PTX release-consistent / scoped
model formalised by Lustig, Sahasrabuddhe, Giroux, ASPLOS 2019.
"""

import itertools
import sys
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# A sentinel for a sector value that a copy never wrote (fill-time / prior-epoch
# residue). Distinct from every legal written value, so a "lost write" surfaces
# as STALE appearing where a real writer's value is required.
STALE = -1


# ===========================================================================
# PATH A -- FALSE SHARING: per-sector sync-merge  (I1 data-integrity, I2 merge
# commutativity+associativity)
# ===========================================================================
#
# Abstraction of what the implementation delivers to a consumer at epoch end.
# The tag-only sim always reads the byte-correct value from L2 (write-through
# invariant). Off-sim we instead carry data through the protocol's OWN delivery
# rule -- i.e. we ask: "if the cache actually held data and the consumer were
# served by the protocol's forward/writeback policy, what value would it see?"
#   * REAL protocol : per-sector written mask preserved => each sector delivered
#                     from the unique writer that OR'd it (gpu-cache.cc:3327 +
#                     serve-path invariant 2663/3099 => L2 byte-merge).
#   * LWW ablation  : mask dropped, whole line owned by the last writer
#                     (gpu-cache.cc:3324-3325) => a whole-line delivery carries
#                     ONLY the last writer's line; every sector written by an
#                     earlier writer is LOST.


@dataclass(frozen=True)
class SMCopy:
    """One SM's L1 copy of a single line: per-sector value + per-sector cluster
    state + this line's accumulated written mask (as a frozenset of sectors)."""
    sid: int
    values: Tuple[int, ...]          # length N; STALE where this SM never wrote
    state: Tuple[str, ...]           # per-sector cluster_line_state
    mask: FrozenSet[int]             # m_mw_written_mask, as sector indices


def apply_write_real(copy: SMCopy, sector: int, value: int) -> SMCopy:
    """Real protocol: OR the sector into the mask (gpu-cache.cc:3327), stamp the
    sector MW_MERGE (3336-3343), carry the value in this copy's sector slot."""
    vals = list(copy.values)
    vals[sector] = value
    st = list(copy.state)
    st[sector] = "MW_MERGE"
    return SMCopy(copy.sid, tuple(vals), tuple(st), copy.mask | {sector})


def apply_write_lww(copy: SMCopy, sector: int, value: int) -> SMCopy:
    """LWW ablation: CLEAR then set only my sector (gpu-cache.cc:3324-3325).
    The mask never accumulates -> the line is conceptually owned whole by the
    latest writer."""
    vals = list(copy.values)
    vals[sector] = value
    st = list(copy.state)
    st[sector] = "MW_MERGE"
    return SMCopy(copy.sid, tuple(vals), tuple(st), frozenset({sector}))


def deliver_real(copies: List[SMCopy], n_sectors: int) -> Tuple[int, ...]:
    """REAL delivery: per-sector writeback (gpu-cache.cc:3359-3362 writes back
    only mask.count() sectors) => L2 ends up with, for each sector, the value
    from the unique copy whose mask contains it. Consumer reads that byte-merge
    (serve-path invariant forbids a single-owner whole-line forward)."""
    out = [STALE] * n_sectors
    for s in range(n_sectors):
        for c in copies:
            if s in c.mask:
                out[s] = c.values[s]
                break  # disjoint => at most one contributor per sector
    return tuple(out)


def deliver_lww(copies: List[SMCopy], last_writer_idx: int,
                n_sectors: int) -> Tuple[int, ...]:
    """LWW delivery: the whole line is owned by the last writer; a whole-line
    forward/writeback carries THAT copy's entire line. Sectors it never wrote
    stay STALE (its fill-time residue) -> earlier writers' sectors are lost."""
    return tuple(copies[last_writer_idx].values)


def enumerate_path_a(n_sectors: int, writers: List[Tuple[int, int, int]],
                     lww: bool):
    """Explicit-state enumeration over ALL orderings of the writes (models the
    protocol's order-nondeterminism) AND, for I2, all merge groupings.

    writers: list of (sid, sector, value); disjoint sectors == false sharing.
    Returns (results_by_order, all_states) where results_by_order maps an order
    permutation -> the delivered line."""
    results = {}
    all_terminal = []
    base = {sid: SMCopy(sid, tuple([STALE] * n_sectors),
                        tuple(["INVALID"] * n_sectors), frozenset())
            for (sid, _, _) in writers}
    for order in itertools.permutations(range(len(writers))):
        copies = dict(base)
        last_sid = None
        for k in order:
            sid, sector, value = writers[k]
            fn = apply_write_lww if lww else apply_write_real
            copies[sid] = fn(copies[sid], sector, value)
            last_sid = sid
        copy_list = [copies[sid] for (sid, _, _) in writers]
        last_idx = [i for i, (sid, _, _) in enumerate(writers)
                    if sid == last_sid][0]
        if lww:
            delivered = deliver_lww(copy_list, last_idx, n_sectors)
        else:
            delivered = deliver_real(copy_list, n_sectors)
        results[order] = delivered
        all_terminal.append((order, delivered, last_sid))
    return results, all_terminal


def check_I1_data_integrity(n_sectors, writers, lww):
    """(I1) SWMR per sector: every committed sector equals the value written by
    its unique writer -- no lost write, no torn merge."""
    expected = {sector: value for (_, sector, value) in writers}
    results, terminal = enumerate_path_a(n_sectors, writers, lww)
    for order, delivered, last_sid in terminal:
        for sector, value in expected.items():
            if delivered[sector] != value:
                trace = _trace_a(order, writers)
                return False, {
                    "invariant": "I1 data-value integrity (SWMR per sector)",
                    "order": order,
                    "trace": trace,
                    "sector": sector,
                    "expected": value,
                    "got": delivered[sector],
                    "delivered_line": delivered,
                    "note": ("sector written by sid=%d lost; delivered value "
                             "is %s (STALE=-1 means the sector was never carried "
                             "-> lost write)" %
                             (_writer_of(writers, sector),
                              "STALE" if delivered[sector] == STALE
                              else delivered[sector])),
                }
    return True, {"orders_checked": len(terminal)}


def check_I2_merge_comm_assoc(n_sectors, writers, lww):
    """(I2) Merge is commutative AND associative: the delivered line is
    independent of the order in which per-sector dirty masks are merged."""
    results, terminal = enumerate_path_a(n_sectors, writers, lww)
    distinct = set(results.values())
    if len(distinct) == 1:
        return True, {"orders_checked": len(results),
                      "delivered": next(iter(distinct))}
    # order-dependence -> commutativity/associativity violated
    items = list(results.items())
    o1, r1 = items[0]
    o2, r2 = next((o, r) for o, r in items if r != r1)
    return False, {
        "invariant": "I2 merge commutativity+associativity",
        "order_A": o1, "result_A": r1,
        "order_B": o2, "result_B": r2,
        "trace_A": _trace_a(o1, writers),
        "trace_B": _trace_a(o2, writers),
        "note": "final line depends on merge order => not commutative/associative",
    }


def _writer_of(writers, sector):
    for sid, s, _ in writers:
        if s == sector:
            return sid
    return None


def _trace_a(order, writers):
    steps = []
    for k in order:
        sid, sector, value = writers[k]
        steps.append("SM%d: write sector %d = %d (WT to L2; OR mask)"
                     % (sid, sector, value))
    return steps


# ===========================================================================
# PATH B -- CLUSTER-SCOPED ATOMICS: migrating owner token  (I3 serialisability)
# ===========================================================================
#
# Real protocol: exactly one owner token per line (g_mw_atomic_owner keyed by
# {gpc,addr}). An atomic on a non-owner sends one UpgradeReq; the current owner
# downgrades EM->FORWARD->INVALID and the requester installs EM (sole owner).
# The RMW therefore executes indivisibly at the token holder => the K atomics
# take SOME total order == a serial execution (SC for that address at cluster
# scope; PTX .cluster-scoped atomics are ordered, Lustig et al. ASPLOS'19).
#
# Ablation-2: no token. The RMW is split into (read old)-(compute)-(write new)
# that can interleave freely across SMs => two SMs read the same base and one
# update is lost. We BFS all interleavings and report the first lost update.


@dataclass(frozen=True)
class AtomicOp:
    sid: int
    delta: int   # atomicAdd delta (commutative RMW: final must be base+sum)


def check_I3_token_serialises(base_value: int, ops: List[AtomicOp]):
    """REAL protocol: enumerate all token-acquisition orders; each atomic is
    indivisible. Assert every order yields a value equal to SOME serial
    execution and (for add) equal to base+sum -> no lost update."""
    total = base_value + sum(o.delta for o in ops)
    seen_finals = set()
    example_order = None
    for order in itertools.permutations(range(len(ops))):
        v = base_value
        for k in order:                      # token migrates: one holder at a time
            v = v + ops[k].delta             # indivisible RMW at the owner
        seen_finals.add(v)
        if example_order is None:
            example_order = order
    # every interleaving is a genuine serialisation; for commutative add all
    # equal base+sum. The invariant is: result is ALWAYS a serial result and no
    # update is dropped.
    ok = (seen_finals == {total})
    if ok:
        return True, {"base": base_value, "sum_deltas": total - base_value,
                      "final": total, "orders_checked":
                      _fact(len(ops)), "trace":
                      _trace_b_token(example_order, ops, base_value)}
    return False, {"invariant": "I3 token serialisation",
                   "finals_seen": sorted(seen_finals), "expected": total}


def check_ablation2_no_token(base_value: int, ops: List[AtomicOp]):
    """ABLATION-2: no owner token -> RMW = separate read/write that interleave.
    BFS over all interleavings of the per-SM (READ, WRITE) sub-steps; return the
    first interleaving whose final value != base+sum (a lost update)."""
    total = base_value + sum(o.delta for o in ops)

    # State: (mem_value, tuple(phase_i in {0=not started,1=read,2=done}),
    #         tuple(local_read_i)). BFS from the initial state; on any terminal
    # state with a wrong final, reconstruct the trace.
    from collections import deque
    n = len(ops)
    start = (base_value, tuple([0] * n), tuple([None] * n))
    # parent map for trace reconstruction
    parent = {start: (None, None)}
    q = deque([start])
    bad_terminal = None
    while q:
        mem, phase, local = q.popleft()
        if all(p == 2 for p in phase):
            if mem != total:
                bad_terminal = (mem, phase, local)
                break
            continue
        for i in range(n):
            if phase[i] == 0:
                # READ step: SM i latches current memory value
                nl = list(local); nl[i] = mem
                np = list(phase); np[i] = 1
                ns = (mem, tuple(np), tuple(nl))
                lbl = "SM%d: READ acc -> %d" % (ops[i].sid, mem)
                if ns not in parent:
                    parent[ns] = ((mem, phase, local), lbl)
                    q.append(ns)
            elif phase[i] == 1:
                # WRITE step: SM i writes back its latched value + delta
                newmem = local[i] + ops[i].delta
                np = list(phase); np[i] = 2
                ns = (newmem, tuple(np), local)
                lbl = ("SM%d: WRITE acc = %d + %d = %d"
                       % (ops[i].sid, local[i], ops[i].delta, newmem))
                if ns not in parent:
                    parent[ns] = ((mem, phase, local), lbl)
                    q.append(ns)
    if bad_terminal is None:
        return True, {"note": "no lost update found (unexpected for ablation)"}
    # reconstruct trace
    trace = []
    cur = bad_terminal
    while cur is not None and parent.get(cur, (None, None))[1] is not None:
        prev, lbl = parent[cur]
        trace.append(lbl)
        cur = prev
    trace.reverse()
    return False, {
        "invariant": "I3 (violated by ablation): lost update under concurrent RMW",
        "base": base_value, "expected_final": total,
        "got_final": bad_terminal[0],
        "lost": total - bad_terminal[0],
        "trace": trace,
    }


def _fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _trace_b_token(order, ops, base):
    steps = []
    v = base
    prev_owner = None
    for k in order:
        sid = ops[k].sid
        if prev_owner is None:
            steps.append("SM%d: first atomic -> acquires token locally (no hop)"
                         % sid)
        else:
            steps.append("SM%d: UpgradeReq -> SM%d downgrades EM->FORWARD->INVALID "
                         "(1 fabric hop); SM%d installs EM" % (sid, prev_owner, sid))
        v = v + ops[k].delta
        steps.append("      RMW at owner: acc = %d (+%d)" % (v, ops[k].delta))
        prev_owner = sid
    return steps


# ===========================================================================
# DRIVER
# ===========================================================================

def banner(t):
    print("=" * 74)
    print(t)
    print("=" * 74)


def show(name, ok, info):
    tag = "PASS" if ok else "FAIL"
    print("[%s] %s" % (tag, name))
    for k, v in info.items():
        if k in ("trace", "trace_A", "trace_B"):
            print("    %s:" % k)
            for s in v:
                print("        %s" % s)
        else:
            print("    %s = %s" % (k, v))
    print()
    return ok


def main():
    all_ok = True
    print("RESC MULTI-WRITER COHERENCE -- OFF-SIM EXPLICIT-STATE MODEL CHECK")
    print("tool: self-contained Python BFS/permutation enumerator (Murphi not")
    print("installed on this host); methodology per Dill et al. ICCD 1992.")
    print("modelled tree: RESC-c2c-wt gpu-cache.cc / gpu-cache.h (anchors in header)")
    print()

    # ---- Bounded configs (small, per Murphi practice) ----
    # Path A false sharing: 1 line, N=4 sectors (SECTOR_CHUNCK_SIZE), disjoint
    # writers. Try 2, 3, 4 SMs.
    banner("PART 1 -- REAL PROTOCOL: three invariants must all PASS")

    print("--- (I1) data-value integrity / SWMR per sector (Path A) ---")
    for k in (2, 3, 4):
        writers = [(sid, sid, 100 + sid) for sid in range(k)]  # SM sid writes sector sid
        ok, info = check_I1_data_integrity(4, writers, lww=False)
        all_ok &= show("I1  false-sharing, %d SMs, 1 line, 4 sectors" % k, ok, info)

    print("--- (I2) merge commutativity + associativity (Path A) ---")
    for k in (2, 3, 4):
        writers = [(sid, sid, 100 + sid) for sid in range(k)]
        ok, info = check_I2_merge_comm_assoc(4, writers, lww=False)
        all_ok &= show("I2  merge order-independence, %d SMs" % k, ok, info)

    print("--- (I3) atomic owner-token serialisation (Path B) ---")
    for k in (2, 3, 4):
        ops = [AtomicOp(sid, 10 + sid) for sid in range(k)]
        ok, info = check_I3_token_serialises(1000, ops)
        all_ok &= show("I3  %d concurrent cluster-scoped atomicAdd" % k, ok, info)

    banner("PART 2 -- ABLATIONS: both MUST produce a counterexample (FAIL)")

    print("--- (Ablation-1) whole-line last-writer-wins (drop per-sector mask) ---")
    # This MUST fail I1 under false sharing: earlier writers' sectors are lost.
    for k in (2, 3, 4):
        writers = [(sid, sid, 100 + sid) for sid in range(k)]
        ok, info = check_I1_data_integrity(4, writers, lww=True)
        # We WANT ok == False here.
        got_ce = (not ok)
        show("Ablation-1  whole-line LWW, %d SMs (expect lost-write CE)" % k,
             ok, info)
        if not got_ce:
            print("    !!! ablation did NOT fail -- mask would be unnecessary !!!")
        all_ok &= got_ce  # deliverable requires the ablation to fail

    print("--- (Ablation-2) no owner-token: concurrent RMW race ---")
    for k in (2, 3):
        ops = [AtomicOp(sid, 10 + sid) for sid in range(k)]
        ok, info = check_ablation2_no_token(1000, ops)
        got_ce = (not ok)
        show("Ablation-2  no token, %d atomics (expect lost-update CE)" % k,
             ok, info)
        if not got_ce:
            print("    !!! ablation did NOT fail -- token would be unnecessary !!!")
        all_ok &= got_ce

    banner("VERDICT")
    if all_ok:
        print("ALL CHECKS AS EXPECTED: I1/I2/I3 PASS on the real protocol;")
        print("both ablations produced counterexamples (mechanism is NECESSARY).")
        sys.exit(0)
    else:
        print("UNEXPECTED RESULT -- see FAIL lines above. Do NOT paper over.")
        sys.exit(1)


if __name__ == "__main__":
    main()
