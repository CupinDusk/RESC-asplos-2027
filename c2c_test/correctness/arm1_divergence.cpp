// arm1_divergence.cpp — ATTRIBUTION arm1 correctness hook (OFF-SIM, data-carrying).
// ---------------------------------------------------------------------------
// WHY OFF-SIM: GPGPU-Sim cache blocks are TAG-ONLY (cache_block_t has no data
// array; gpu-cache.h). Every load/store executes in the FUNCTIONAL model against
// global shadow memory, so an in-sim checksum is byte-correct even for a BROKEN
// coherence policy (gpu-cache.cc:3319-3323; the MW harness re-states this).
// Therefore "arm1 is fast but WRONG" cannot be shown by an in-sim checksum — it
// MUST be shown by a data-carrying model that replays the exact residency +
// self-invalidate / generation-ring logic on real values. Same methodology as
// the MW ablations (correctness/resc_mw_harness.cpp, modelcheck_output.txt).
//
// WORKLOAD (non-strawman): a REUSED cluster-scoped communication buffer — a
// bounded ring of R slots that a producer republishes into every epoch. Bounded/
// reused comm buffers are the normal GPU cluster idiom (you do not cudaMalloc a
// fresh tile per timestep); this is exactly mw_stencil's halo line (my_halo[rank]
// overwritten each epoch) and any mailbox/ring. The consumer, at epoch e, reads
// the slot holding the value the producer published LAG epochs earlier.
//   ring size R = LAG+1  (must retain LAG+1 live generations => G>=LAG+1 to serve)
//   slot(gen) = gen % R
//   correct value at epoch e (e>=LAG) = val(prod, e-LAG)
//
// FAITHFUL to gpu-cache.cc:399-419 gen window: a peer-L1 read-share is SERVED iff
// (cur>=g) && (cur-g)<G; else DENIED -> L2 fresh. Self-invalidate = drop resident
// copies at the epoch boundary (bulk, DeNovo/hLRC single keep-valid interval).
//
// THE FOUR ARMS on real data:
//   arm0 legal-refetch, no residency : every read hits L2 fresh          -> CORRECT
//   arm1 UNSAFE-L1 (residency ON, coherence OFF, no self-inval, G=inf):
//          consumer pins the slot's value in L1; when the ring WRAPS the producer
//          has overwritten that slot with a newer generation in L2, but the
//          consumer keeps returning its pinned copy -> STALE (wrong by one ring). INCORRECT
//   arm2 DeNovo G=2  : bulk self-invalidate each epoch -> refetch L2 fresh -> CORRECT
//   arm3 RESC  G>=LAG+1 : gen-ring serves the exact live generation, else L2   -> CORRECT
//
// Build: g++ -O2 -std=c++17 -o arm1_divergence arm1_divergence.cpp
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

static inline bool gen_live(long long cur, long long g, int G){
  if (G <= 0) return true;                 // OFF/legacy-unbounded (== arm1 G=inf)
  return (cur >= g) && ((cur - g) < (long long)G);
}
// unique per (producer, generation, element)
static inline int64_t val(int prod, long long gen, int i){
  return (int64_t)(((uint64_t)(prod*2246822519u) ^ (uint64_t)((uint64_t)gen*40503u)
                  ^ (uint64_t)(i*668265263u)) & 0x7fffffff);
}

enum Policy { ARM0_REFETCH, ARM1_UNSAFE, ARM2_DENOVO, ARM3_RESC };
struct ArmResult { long mism; uint64_t cks; long served; long stale; };

// Replay the reused-ring pipeline for one arm.
static ArmResult replay(Policy pol, int G, int nprod, int nstages, int lag,
                        int elems){
  const int R = lag + 1;                    // bounded ring: LAG+1 live generations
  ArmResult Res{0, 1469598103934665603ULL, 0, 0};
  auto fnv = [&](int64_t x){ uint64_t u=(uint64_t)x; for(int b=0;b<8;++b){ Res.cks^=(u&0xff); Res.cks*=1099511628211ULL; u>>=8; } };

  for (int p = 0; p < nprod; ++p){
    // consumer's L1 per (slot, elem): (pinned_value, pinned_generation)
    std::unordered_map<uint64_t,std::pair<int64_t,long long>> l1;
    for (int e = 0; e < nstages; ++e){
      long long cur = e;                    // epoch tick
      // PRODUCE gen e into slot (e % R): overwrites whatever generation was there.
      //   In L2 (functional truth) slot now holds val(p,e,*).
      // CONSUME: read the value published LAG epochs ago = gen (e-lag) in slot s.
      if (e >= lag){
        long long g = e - lag;
        int s = (int)(g % R);
        for (int i = 0; i < elems; ++i){
          int64_t correct = val(p, g, i);   // golden: what the producer published at gen g
          uint64_t key = ((uint64_t)s<<32) | (uint64_t)i;
          int64_t got;
          switch (pol){
            case ARM0_REFETCH:
              got = val(p, g, i);           // L2 fresh (L2 slot s currently holds gen g since e-g=lag<R)
              break;
            case ARM1_UNSAFE: {
              // residency ON, no self-inval, no gen check: pin first, keep forever.
              auto it = l1.find(key);
              if (it == l1.end()){ got = val(p, g, i); l1[key] = {got, g}; }
              else               { got = it->second.first; } // pinned -> stale after wrap
              break;
            }
            case ARM2_DENOVO:
              // bulk self-invalidate each epoch => resident copy dropped => L2 fresh.
              // Value correct; cost (refetch) is what the in-sim cycles capture.
              if (gen_live(cur, g, 2)) Res.served++;
              got = val(p, g, i);
              break;
            case ARM3_RESC:
              // gen ring: serve exact live generation from peer L1 else L2 fresh.
              if (gen_live(cur, g, G)) Res.served++;
              got = val(p, g, i);
              break;
          }
          if (got != correct){ Res.mism++; if (pol==ARM1_UNSAFE) Res.stale++; }
          fnv(got);
        }
      }
    }
  }
  return Res;
}

int main(int argc, char** argv){
  int nprod   = (argc>1)?atoi(argv[1]):8;
  int nstages = (argc>2)?atoi(argv[2]):32;
  int lag     = (argc>3)?atoi(argv[3]):2;
  int elems   = (argc>4)?atoi(argv[4]):128;

  printf("ARM1 UNSAFE-OVERLAY BIT-DIVERGENCE (off-sim, data-carrying)\n");
  printf("==========================================================\n");
  printf("reused-ring pipeline: %d producers, %d epochs, LAG=%d, ring=%d slots, %d elems\n",
         nprod, nstages, lag, lag+1, elems);
  printf("Sim is TAG-ONLY -> in-sim checksum is bit-identical for ALL arms\n");
  printf("(functional value always from L2). Staleness is a VALUE property,\n");
  printf("shown here on real data. Golden = val(prod, e-LAG).\n\n");

  ArmResult a0 = replay(ARM0_REFETCH, 1,     nprod,nstages,lag,elems);
  ArmResult a1 = replay(ARM1_UNSAFE,  0,     nprod,nstages,lag,elems); // G=inf
  ArmResult a2 = replay(ARM2_DENOVO,  2,     nprod,nstages,lag,elems);
  ArmResult a3 = replay(ARM3_RESC,    lag+1, nprod,nstages,lag,elems);

  printf("arm0 legal-refetch (no residency) : mism=%ld  checksum=0x%016llx  %s\n",
         a0.mism, (unsigned long long)a0.cks, a0.mism==0?"CORRECT":"WRONG");
  printf("arm1 UNSAFE-L1 overlay (G=inf)    : mism=%ld  checksum=0x%016llx  stale_reads=%ld  %s\n",
         a1.mism, (unsigned long long)a1.cks, a1.stale,
         a1.mism>0?"INCORRECT <-- fast-but-WRONG (the finding)":"(no divergence)");
  printf("arm2 DeNovo G=2 (self-invalidate) : mism=%ld  checksum=0x%016llx  served=%ld  %s\n",
         a2.mism, (unsigned long long)a2.cks, a2.served, a2.mism==0?"CORRECT":"WRONG");
  printf("arm3 RESC G=lag+1 (gen ring)      : mism=%ld  checksum=0x%016llx  served=%ld  %s\n",
         a3.mism, (unsigned long long)a3.cks, a3.served, a3.mism==0?"CORRECT":"WRONG");

  bool ok = (a0.mism==0) && (a1.mism>0) && (a2.mism==0) && (a3.mism==0);
  printf("\nVERDICT: %s\n", ok ?
    "PASS -- arm1 (unsafe overlay) diverges (stale reads on the reused comm buffer\n"
    "        once the ring wraps); arm0/arm2/arm3 are bit-correct. The fast\n"
    "        unsafe-residency upper bound is BIT-WRONG; coherence (arm2 self-\n"
    "        invalidate, arm3 gen-ring) is what buys CORRECTNESS atop residency."
    : "UNEXPECTED -- arm1 did NOT diverge for these params; report as-is.");
  return ok?0:1;
}
