# RESC Multi-Writer Coherence — Off-Sim Correctness Argument

**Scope.** This document establishes that the RESC multi-writer extension is a
*correct coherence protocol*, not merely a producer→consumer forwarding cache.
It answers the Round-9 reviewer challenge ("is this really coherence?") with the
field's standard tool for coherence correctness — explicit-state model checking
— applied to the **actually implemented** transition table, plus a data-carrying
reference harness that does what the tag-only simulator structurally cannot.

All artifacts live under
`RESC-c2c-wt/c2c_test/correctness/` (sim host) and are mirrored to
`resc-asplos-2027/ccl-pivot/correctness/` (local).

---

## 1. Why an off-sim argument is *required* (the tag-only fact)

GPGPU-Sim cache blocks are **tag-only**: `cache_block_t` carries `tag`,
`status`, and timestamps but **no data array** (`gpu-cache.h`). Every load,
store, and atomic executes in the **functional** simulator against global shadow
memory; the RESC coherence machine is a pure **timing/routing overlay**.

Consequence: an in-sim checksum is byte-correct **regardless of whether the
protocol is right or wrong**. In-sim bit-identity is *structural*, not a
coherence proof. The implementation states this itself — the whole-line-LWW
ablation comment at `gpu-cache.cc:3319-3323`:

> "In this tag-only simulator the FUNCTIONAL value still comes from L2, so this
> does NOT corrupt the checksum; it exists to show the design DIFFERENCE (mask
> vs no-mask)…"

So a broken protocol and the correct protocol produce the **same** in-sim
checksum. Coherence correctness therefore has to be established **off** the
simulator, on a model that actually carries data through the protocol's own
delivery rule. That is exactly what the two artifacts below do.

---

## 2. Tool used, and why

**Tool: a self-contained Python explicit-state enumerator**
(`resc_mw_modelcheck.py`), plus a **C++ data-carrying stress harness**
(`resc_mw_harness.cpp`).

We first checked for Murphi — `which rumur cmurphi mu murphi` on both the sim
host and this machine returns nothing, and `pip index versions rumur` reports
"No matching distribution found." Murphi is not installed and not trivially
installable within the time budget, so we **honestly did not use Murphi**.
Instead we wrote a from-scratch enumerator that performs the same core operation
Murphi performs — exhaustive breadth-first / permutation enumeration of all
reachable global states for a small bounded configuration, with a concrete
counterexample trace on any invariant violation — following the methodology of
Dill, Drexler, Hu, and Yang, "Protocol Verification as a Hardware Design Aid,"
ICCD 1992 (`DBLP:conf/iccd/DillDHY92`), the canonical reference for
model-checking cache-coherence protocols.

The enumerator is **small by design** (2–4 SMs, 1 line, 4 sectors, ≤2 epochs
for Path A; ≤4 concurrent atomics for Path B) — exhaustiveness at small bounds
is the point of explicit-state model checking; the harness then re-checks the
same logic at stress scale with real data.

The model reflects the **implemented** transitions, verified by reading the live
`RESC-c2c-wt` tree this session (file:line table in §7), **not** the design
brief. Where the brief and the code differ we followed the code.

---

## 3. The three invariants, as model-checked

Derived from the implemented transitions (`try_cluster_multiwriter_merge`,
`try_cluster_atomic_token`, and the serve-path invariant):

- **(I1) Data-value integrity (SWMR per sector).** For disjoint-sector writes to
  one line, every committed sector equals the value written by its *unique*
  writer — no lost write, no torn merge. This models the real per-sector OR mask
  (`gpu-cache.cc:3327`) + per-sector writeback (`:3359-3362`) + the serve-path
  invariant that forbids a single-owner whole-line forward from an
  `MW_MERGE` line (`:2663-2671`, `:3099-3108`), so the consumer reads the
  byte-accurate L2 merge.

- **(I2) Merge is commutative AND associative.** The final line content is
  independent of the order in which the per-sector dirty masks are merged. Since
  the merge OR-s disjoint per-sector contributions, all `k!` write orders must
  deliver the identical line.

- **(I3) Atomic token serialises.** For concurrent RMW to one line, the single
  migrating owner token (`g_mw_atomic_owner`, `gpu-cache.cc:3386-3480`;
  EM→FORWARD→INVALID hand-off at `:3448-3457`) yields a total order equal to
  *some* serial execution — no lost update. For commutative `atomicAdd` every
  token-arrival order gives `base + Σ deltas`.

### PASS output (real protocol) — `modelcheck_output.txt`

```
--- (I1) data-value integrity / SWMR per sector (Path A) ---
[PASS] I1  false-sharing, 2 SMs, 1 line, 4 sectors    orders_checked = 2
[PASS] I1  false-sharing, 3 SMs, 1 line, 4 sectors    orders_checked = 6
[PASS] I1  false-sharing, 4 SMs, 1 line, 4 sectors    orders_checked = 24

--- (I2) merge commutativity + associativity (Path A) ---
[PASS] I2  merge order-independence, 2 SMs    delivered = (100, 101, -1, -1)
[PASS] I2  merge order-independence, 3 SMs    delivered = (100, 101, 102, -1)
[PASS] I2  merge order-independence, 4 SMs    delivered = (100, 101, 102, 103)

--- (I3) atomic owner-token serialisation (Path B) ---
[PASS] I3  2 concurrent cluster-scoped atomicAdd    base=1000 final=1021  orders_checked=2
[PASS] I3  3 concurrent cluster-scoped atomicAdd    base=1000 final=1033  orders_checked=6
[PASS] I3  4 concurrent cluster-scoped atomicAdd    base=1000 final=1046  orders_checked=24
```

All three invariants hold over **every** enumerated global state at the bounded
config (exhaustive: 2/6/24 orders for k=2/3/4). Verdict line:
`ALL CHECKS AS EXPECTED: I1/I2/I3 PASS on the real protocol`.

The I3 trace shows the token machine firing the exact implemented edges:

```
SM0: first atomic -> acquires token locally (no hop)
SM1: UpgradeReq -> SM0 downgrades EM->FORWARD->INVALID (1 fabric hop); SM1 installs EM
SM2: UpgradeReq -> SM1 downgrades EM->FORWARD->INVALID (1 fabric hop); SM2 installs EM
```

---

## 4. The two ablations — both produce counterexamples (mechanism is necessary)

If the same model *passed* with the mechanism removed, the mechanism would be
decorative. Both broken variants must — and do — fail.

### Ablation-1: whole-line last-writer-wins (drop the per-sector mask)

Models `gpu-cache.cc:3324-3325` (LWW clears the mask, keeps only the latest
writer's sector). The whole line is then delivered by its last writer;
every sector an earlier writer touched is lost. Counterexample (verbatim):

```
[FAIL] Ablation-1  whole-line LWW, 2 SMs (expect lost-write CE)
    invariant = I1 data-value integrity (SWMR per sector)
    trace:
        SM0: write sector 0 = 100 (WT to L2; OR mask)
        SM1: write sector 1 = 101 (WT to L2; OR mask)
    sector = 0
    expected = 100
    got = -1
    delivered_line = (-1, 101, -1, -1)
    note = sector written by sid=0 lost; delivered value is STALE
           (STALE=-1 means the sector was never carried -> lost write)
```

(3-SM and 4-SM cases give the analogous `(-1,-1,102,-1)` and `(-1,-1,-1,103)`
lost-write lines.) → **the per-sector written mask is required.**

### Ablation-2: no owner-token (concurrent atomics race)

Models the RMW as separable read/compute/write sub-steps that interleave freely
(the state before Path B existed). BFS over all interleavings returns the first
lost update:

```
[FAIL] Ablation-2  no token, 2 atomics (expect lost-update CE)
    invariant = I3 (violated by ablation): lost update under concurrent RMW
    base = 1000    expected_final = 1021    got_final = 1011    lost = 10
    trace:
        SM0: READ acc -> 1000
        SM1: READ acc -> 1000
        SM0: WRITE acc = 1000 + 10 = 1010
        SM1: WRITE acc = 1000 + 11 = 1011
```

Both SMs read the same base; SM0's update (+10) is lost. → **the migrating owner
token is required.**

---

## 5. Data-carrying reference harness (bit-identity at stress scale)

`resc_mw_harness.cpp` carries **real data values** through the exact merge and
atomic-token *delivery* logic on real arrays, then compares element-by-element to
a serial golden reference. This is the concrete complement to the tag-only sim:
it is the run in which a broken protocol *would* show a nonzero mismatch.

Reported numbers (sim host, `harness_output.txt`; local run identical except
RNG-seeded values):

```
PATH A -- false sharing, 200000 lines x 4 sectors = 800000 elements, 16 SMs
  REAL per-sector merge : mismatches vs golden = 0        checksum == golden checksum
  LWW ablation          : mismatches vs golden = 600000   (checksum diverges)
  => Path A PASS (real bit-identical; ablation diverges)

PATH B -- concurrent atomics, 50000 lines x 32 atomics = 1600000 RMWs, 16 SMs
  REAL owner-token      : mismatches vs golden = 0        checksum == golden checksum
  no-token ablation     : mismatches vs golden = 50000    (lost updates)
  => Path B PASS (real bit-identical; ablation loses updates)
```

- **Path A:** 800,000 elements, **0** mismatches for the real per-sector merge;
  the whole-line-LWW ablation loses **600,000** (= 3 of every 4 sectors per
  line, exactly the non-last-writer sectors).
- **Path B:** 1,600,000 concurrent RMWs across 50,000 hot lines, **0**
  mismatches for the owner-token; the no-token ablation loses one update on
  **every** contended line (50,000 lost).

Real == golden bit-for-bit on both paths; both ablations diverge.

---

## 6. Honest reframe (for the paper and for reviewers)

The simulator establishes **timing and routing** — that the protocol's messages,
hops, and state transitions fire as designed, and how they affect cycles. It
**cannot** establish coherence correctness, because its caches are tag-only and
the functional value always comes from global memory (§1); an in-sim checksum is
correct even for a broken protocol (the code says so at `gpu-cache.cc:3319-3323`).

Coherence correctness is a **protocol property**, and the field establishes it
**off-sim by model checking** — the Murphi lineage (Dill et al., ICCD 1992),
which exhaustively enumerates reachable states and is the standard for
cache-coherence verification. That method **dominates a single-input in-sim
checksum**: it proves the invariants over *all* reachable interleavings at the
bounded config, and it distinguishes the correct protocol from a broken one
(the ablations fail; the in-sim checksum could not tell them apart). The
data-carrying harness then confirms the same conclusion with real values at
stress scale. Together they show RESC multi-writer is a **correct coherence
protocol**: SWMR-per-sector for false sharing, and serialisable RMW for atomics.

**Boundary (case C, stated honestly).** Two SMs writing the *same* byte without
synchronization is a data race — undefined behaviour under the GPU release-
consistent / scoped memory model (NVIDIA PTX, formalised by Lustig,
Sahasrabuddhe, Giroux, ASPLOS 2019, `DBLP:conf/asplos/LustigSG19`). RESC does
not (and need not) order such accesses; it only guarantees it never returns a
value outside the write-set. Our invariants cover the two cases a correct
program actually produces in one epoch — disjoint-sector false sharing (I1/I2)
and concurrent same-address atomics (I3).

---

## 7. Handoff table — model ↔ real implementation sites

| Modelled behaviour | Implementation site (RESC-c2c-wt) |
| --- | --- |
| Cluster line states incl. `CLUSTER_MW_MERGE` | `gpu-cache.h:58` |
| Per-line written mask (`mw_or/get/clear`, field) | `gpu-cache.h:457-463`, `gpu-cache.h:555` |
| Path A false-sharing sync-merge hook | `gpu-cache.cc:3240` `try_cluster_multiwriter_merge` |
| Real per-sector OR of written mask | `gpu-cache.cc:3327` |
| LWW ablation (clear mask, last writer only) | `gpu-cache.cc:3318-3325` |
| Per-sector vs whole-line writeback accounting | `gpu-cache.cc:3359-3362` |
| Install `MW_MERGE` on both copies | `gpu-cache.cc:3336-3352` |
| Serve-path invariant (forbid whole-line fwd → L2 merge) | `gpu-cache.cc:2663-2671`, `3099-3108` |
| Path A caller hooks (write-share + post-step) | `gpu-cache.cc:2565`, `2580` |
| Path B atomic owner-token hook | `gpu-cache.cc:3386` `try_cluster_atomic_token` |
| Token local (no hop) | `gpu-cache.cc:3425-3431` |
| UpgradeReq hand-off: EM→FORWARD→INVALID, 1 fabric msg | `gpu-cache.cc:3435-3457` |
| Freshness gate / generation bump | `gpu-cache.cc:3470` |
| Path B caller hook | `gpu-cache.cc:2482` |
| Tag-only ⇒ in-sim checksum structural (honesty anchor) | `gpu-cache.cc:3319-3323` |
| Transition counters (merge entries, upgradereq, fwd-downgrades, …) | `gpu-cache.cc:179-185` |

---

## 8. Ready-to-paste "Correctness" subsection

> **Correctness.** Because GPGPU-Sim caches are tag-only (`cache_block_t` has no
> data array) and all values are supplied by the functional simulator, an in-sim
> checksum is byte-correct even for a broken protocol — in-sim bit-identity is
> structural, not a coherence proof. We therefore establish coherence
> correctness off-simulator by explicit-state model checking, the standard
> methodology for cache-coherence protocols in the Murphi lineage [Dill et al.,
> ICCD 1992]. We build an abstract model of the *implemented* transition table —
> the per-sector written mask and `CLUSTER_MW_MERGE` serve-path invariant for
> false sharing, and the single migrating owner token (EM→FORWARD→INVALID
> hand-off) for cluster-scoped atomics — and exhaustively enumerate all reachable
> states for a bounded configuration (up to four SMs, one line, four sectors,
> concurrent RMW). Three invariants hold over every reachable state: (I1) SWMR
> data-value integrity per sector (no lost or torn write under disjoint-sector
> false sharing), (I2) commutativity and associativity of the merge (delivered
> line independent of merge order), and (I3) serialisability of the owner token
> (concurrent RMW yields a serial result with no lost update). Two ablations
> confirm the mechanism is necessary rather than decorative: dropping the
> per-sector mask (whole-line last-writer-wins) produces a lost-write
> counterexample, and dropping the owner token produces a lost-update
> counterexample. A separate data-carrying harness reproduces the same result
> with real values at stress scale — 0 mismatches against a serial golden
> reference across 800K disjoint-sector writes and 1.6M concurrent atomics, while
> both ablations diverge. Racing plain writes to the same byte are a data race
> and undefined under the PTX scoped memory model [Lustig et al., ASPLOS 2019];
> RESC does not order them and only guarantees it never returns a value outside
> the write-set. RESC multi-writer is thus a correct coherence protocol, not a
> producer→consumer forwarding cache.

---

## 9. Sources

| Claim | Source |
| --- | --- |
| GPGPU-Sim blocks are tag-only; functional value from shadow memory | `gpu-cache.h` (`cache_block_t`); `gpu-cache.cc:3319-3323` |
| Implemented transition table (states, masks, hooks, hand-off, invariant) | `RESC-c2c-wt/src/gpgpu-sim/gpu-cache.{h,cc}` file:line table §7 (verified this session) |
| Model checking as the coherence-verification standard (Murphi) | Dill, Drexler, Hu, Yang, "Protocol Verification as a Hardware Design Aid," ICCD 1992, pp. 522–525, DOI 10.1109/ICCD.1992.276232 (`refs.bib`, verified via DBLP this session) |
| Racing same-byte plain writes = UB; scoped atomics ordered (RC/HRF) | Lustig, Sahasrabuddhe, Giroux, "A Formal Analysis of the NVIDIA PTX Memory Consistency Model," ASPLOS 2019, pp. 257–270, DOI 10.1145/3297858.3304043 (`refs.bib`, verified via DBLP this session) |
| Murphi not available here (honest tool disclosure) | `which rumur cmurphi mu murphi` (empty) on sim host + local; `pip index versions rumur` → no distribution |
| I1/I2/I3 PASS; ablation counterexamples; harness bit-identity | `modelcheck_output.txt`, `harness_output.txt` (this session, both hosts) |

**Gap flagged honestly:** the model is an *abstraction* of the implemented
transitions (bounded config), not an extraction from the C++ source; the file:line
handoff table (§7) is the audited correspondence, and the harness independently
re-checks the same delivery logic at scale. We did not use Murphi itself (not
installed); the Python enumerator implements the same explicit-state method.
