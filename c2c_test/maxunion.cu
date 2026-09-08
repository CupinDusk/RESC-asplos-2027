// maxunion.cu -- THE MAX-UNION ARENA TAX over HETEROGENEOUS RESIDENT TASKS.
//
// ============================================================================
// WHAT THIS MODELS, AND WHAT IT DOES NOT CLAIM
// ============================================================================
// This is a MODEL of the persistent-megakernel structure, NOT a reproduction of
// any real kernel.  It is modelled on the published structure of Stanford /
// HazyResearch Megakernels (repo HazyResearch/Megakernels, commit
// 7309cec801537b61fea3b50d7dfe454a6cde578e), whose relevant properties were
// verified from that source at that SHA:
//
//   * ONE RESIDENT CTA PER SM.  demos/low-latency-llama/llama.cuh:144
//         dim3 grid() { return dim3(sm_count); }
//   * HETEROGENEOUS OPCODES ON THAT ONE CTA.  demos/low-latency-llama/llama.cuh:7-13
//     defines 7 opcodes (RMS_QKV_MatVecRopeAppend, PartialAttention,
//     AttentionReduction, O_ProjResidual, RMS_DoubleMatVecSiLU,
//     DownProjResidual, RMS_LM_Head).
//   * A FIXED ARENA SIZED FOR THE LARGEST TASK TYPE, PASSED AS A PER-LAUNCH
//     SCALAR.  include/config.cuh:42-44
//         PAGE_SIZE = 16384;  NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
//         static_assert(NUM_PAGES == 13, ...)
//     with config.cuh:34-39 giving STATIC_SHARED_MEMORY = 512 +
//     2*(4096 + (32+128)*4 + 32*8) = 10496 B and DYNAMIC_SHARED_MEMORY =
//     227000 - 10496 = 216504 B; 216504/16384 = 13 pages.  It reaches the
//     kernel as ONE launch scalar (llama.cuh:146
//     `int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }`),
//     so it CANNOT be scaled per block or per task item.
//   * THE LARGEST TYPE NEEDS ALL 13 PAGES; THE COMMON TYPE NEEDS 2.
//     matvec: demos/low-latency-llama/matvec_pipeline.cuh:121-122
//         needed_pages = 1 + min(inst.iters, INPUT_PIPELINE_STAGES)*STAGE_PAGES
//     with INPUT_PIPELINE_STAGES=3 (:8) and STAGE_PAGES=4 (:10) => 1+3*4 = 13.
//     attention: demos/low-latency-llama/attention_partial.cu:11-12
//         QOL_PAGE = 0; KV_PAGE = 1;  => 2 pages, and :300-303 explicitly
//     releases laneid >= 2 .. NUM_PAGES, i.e. the other 11 pages are IDLE.
//     Idle-but-reserved = 216504 - 2*16384 = 183736 B = 84.9% of the arena.
//   * RECYCLING EXISTS, IS THE AUTHORS' OWN ESCAPE, AND IS REALLOCATION WITHIN
//     THE FIXED ARENA -- NOT SHRINKAGE.  matvec_pipeline.cuh:78-101 picks a
//     release order that DEPENDS ON inst.iters; attention_partial.cu:280
//     `int ret_order[13] = {2,3,...,12,0,1}`.  Both index a 13-entry order: the
//     arena is still 13 pages.  Recycling changes WHEN a page is reusable
//     (latency), never HOW MANY pages the CTA holds (capacity).
//
// ============================================================================
// THE CLAIM IS THE MAX-UNION TAX.  IT IS *NOT* "RUN-TIME-UNKNOWN SIZE".
// ============================================================================
// The "size is only known at run time" framing is REFUTED by Stanford's own
// host scheduler: iters / block_idxs / num_partials are computed in Python and
// baked into the instruction word
// (megakernels/demos/latency/scheduler.py:155-161, :203-208), so sizes are
// SCHEDULE-TIME known and fixed before launch.  A reviewer can refute that
// framing with the authors' own scheduler.py.  This benchmark therefore models
// only the surviving, defensible half:
//
//     A single resident CTA serves >= 2 task types over its lifetime.  The
//     staging arena must be sized for the LARGEST type it will ever see.  While
//     it runs a small type, the difference is reserved and idle.  Bucketing
//     cannot fix this: there is no launch boundary at which to bucket, because
//     the SAME resident CTA sees both types.
//
// The task mix here is INTERLEAVED for exactly that reason, and ARM 4 MEASURES
// the bucketing claim rather than asserting it (see below).
//
// ============================================================================
// OCCUPANCY IS *NOT* THE CLAIM.  L1 CAPACITY IS.
// ============================================================================
// Stanford is already pinned to 1 CTA/SM by its own arena (227000 B of a
// 227 KB per-block limit; config.cuh:30 MAX_SHARED_MEMORY) and by
// __launch_bounds__.  At 1 CTA/SM, releasing SMEM recovers NO additional CTA.
// So the claim is NOT occupancy.  What SMEM release buys on this hardware is L1
// CACHE CAPACITY: the unified data cache is split between SMEM and L1, so SMEM
// reserved is L1 forgone.  In this simulator that path is
//   shader.cc:3596-3617   assoc = max_assoc * (1 - shmem_option/unified)
//   abstract_hardware_model.cc:856,905   smem = dynamic_smem + static_smem
// so DYNAMIC shared memory does drive the split.  Every arm here pins itself to
// 1 CTA/SM with __launch_bounds__(64,1) so the two effects cannot be confused.
//
// CTAs/SM and SMEM/CTA are PRINTED PER ARM (the MAXUNION line below, plus the
// simulator's own "CTA/core =" line), because this project has hit the
// empty-SMEM artifact three times and a per-arm occupancy figure is the only way
// an arm can prove it is in the regime it claims.
//
// WHY DYNAMIC (not static) SHARED MEMORY IS USED HERE.  A static __shared__
// array that an arm never reads is DELETED by ptxas, which silently produces a
// zero-SMEM arm -- the artifact named above.  A dynamic allocation is a launch
// parameter, so it cannot be dead-code-eliminated, and it is ALSO the faithful
// model: Stanford's arena is a per-launch scalar.
//
// MODELLING CONVENIENCE, DISCLOSED.  The arena is RESERVED at full size but only
// a bounded slice is TOUCHED per task, so simulated cycles do not scale with
// arena bytes.  The capacity cost is charged through k.smem (the L1 split
// above), which is the cost under test; the arena's own compute is not modelled.
// This is the same convention resc_e2e.cu:260-263 already uses in this project,
// and it is disclosed rather than hidden.  Consequence: this benchmark can show
// what the max-union RESERVATION costs, and cannot show what the reserved
// arena's compute would cost.
//
// ============================================================================
// ARMS (selected at compile time by -DARM=n)
// ============================================================================
//   ARM=0  WORST_CASE_ARENA  : dynamic SMEM = ARENA_PAGES*PAGE_BYTES, sized for
//          the LARGEST task type and paid ALWAYS, on every task of every type.
//          Communication is staged through that arena and read by peers with
//          cluster.map_shared_rank -- i.e. DSM, the incumbent.
//   ARM=1  RECYCLING         : identical arena SIZE, but the staging slot is
//          re-assigned across task items (a rotating page offset), exactly the
//          escape the Stanford authors implemented.  It must show the arena does
//          NOT shrink -- same SMEM/CTA, same CTAs/SM, same L1 assoc -- so it
//          changes latency, not capacity.  This arm exists to test the naive
//          objection "recycling already solves it".
//   ARM=2  RESC              : NO SMEM staging reservation at all.  The
//          communicated data lives in global lines read with ld.global.ca and
//          REGISTERED as marked, so RESC's cluster read-share can serve a peer
//          from the producer's L1.  Retention is chosen at RUN TIME by
//          RESC_RETAIN, so ARM 2 and ARM 3 are THE SAME BINARY.
//   ARM=3  RESC_NORETAIN     : the ARM=2 binary with RESC_RETAIN unset, to
//          attribute what RETENTION itself buys as distinct from what the SMEM
//          release buys.
//   ARM=4  BUCKETED_IDEAL    : SMEM staging like ARM 0, but the arena is sized
//          for the SMALL type only -- what software could do IF the resident
//          CTA saw one type.  This SEPARATES the two components of any gap:
//            (ARM0 - ARM4) is the MAX-UNION tax (heterogeneity),
//            (ARM4 - ARM2) is SMEM-staging vs L1 (not max-union).
//          At big_every > 0 this arm is ILLEGAL, and that is MEASURED, not
//          asserted: a large task's payload does not fit the small arena, so the
//          arm's own checksum MISMATCHES and it reports BITIDENTICAL=0.  A
//          BITIDENTICAL=0 on ARM 4 with big tasks present is the DESIGNED
//          outcome and is the computed proof that bucketing is not a legal
//          escape under a heterogeneous mix.  At big_every = 0 it is legal and
//          must report BITIDENTICAL=1.
//
// Args: <num_clusters> <ntasks> <big_every> <comm_lines> <priv_lines>
//   ntasks     = task items each resident CTA pulls from its queue
//   big_every  = every big_every-th item is the LARGE type (A); the rest are the
//                SMALL communicating type (B).  big_every=1 => all large;
//                big_every=0 => all small (the degenerate homogeneous setting).
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8
#endif
#ifndef ARM
#define ARM 2
#endif
// Arena geometry, Stanford's verbatim (config.cuh:42-44): 13 pages of 16384 B
// for the largest type, 2 pages for the common type.
#ifndef ARENA_PAGES
#define ARENA_PAGES 13
#endif
#ifndef SMALL_PAGES
#define SMALL_PAGES 2
#endif
#ifndef PAGE_BYTES
#define PAGE_BYTES 16384
#endif
// Bounded slice actually touched per task, in ints (modelling convenience,
// disclosed in the header).  Applies identically in every arm.
// ADDENDUM 03: raised 512 -> 8192.  At 512 BOTH task types clamped to the same
// 512 ints, so type A and type B did IDENTICAL touched work and the big_every
// (heterogeneity) axis -- the experiment's headline axis -- was inert BY
// CONSTRUCTION.  A flat BE result would then have been an artifact of this cap,
// not a finding.  At 8192 the large type touches 4x the small one at
// comm_lines=64 (8192 vs 2048 ints), so heterogeneity is measurable.  The cap
// still binds on type A (8192 of 13312 ints = 62%), so the arena's FULL compute
// is still NOT modelled -- that limitation is reported with the headline, and
// guard G12 makes an inert axis self-announcing (touched_ratio must be > 1.00).
#ifndef TOUCH_CAP
#define TOUCH_CAP 8192
#endif
#define LINE_BYTES 128
#define INTS_PER_LINE (LINE_BYTES/4)
#define MAX_TASKS 4096

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// How many arena pages THIS ARM reserves.  ARM 2/3 (RESC) reserve NO staging.
#if ARM == 0 || ARM == 1
#define RESERVED_PAGES ARENA_PAGES
#elif ARM == 4
#define RESERVED_PAGES SMALL_PAGES
#else
#define RESERVED_PAGES 0
#endif
#define STAGES_IN_SMEM (ARM == 0 || ARM == 1 || ARM == 4)

// simulator hook: intercepted in cuda-sim/instructions.cc (the same hook
// markfrac.cu and pipe_chain.cu use)
__device__ volatile int g_reg_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_reg_flag = (int)((unsigned long long)addr & 1); (void)addr;
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)LINE_BYTES - 1);
}
__device__ __forceinline__ void register_range_warp0(void* base, size_t bytes){
  uintptr_t b=(uintptr_t)base, start=align_down_line(b);
  int nlines=(int)(((b+bytes-start)+LINE_BYTES-1)/LINE_BYTES);
  int lane=threadIdx.x&31;
  for(int off=0;off<nlines;off+=32){
    int idx=off+lane;
    uintptr_t line=start+(uintptr_t)((idx<nlines)?idx:0)*LINE_BYTES;
    gpgpusim_register_cluster_coherent_addr((void*)line);
  }
}
__device__ __forceinline__ int ld_ca_s32(const int* p){ int v;
  asm volatile("ld.global.ca.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_wb_s32(int* p,int v){
  asm volatile("st.global.wb.s32 [%0],%1;" :: "l"(p),"r"(v) : "memory"); }

__device__ unsigned long long g_checksum = 0;

// deterministic value generators shared host/device so the reference is exact
__host__ __device__ __forceinline__ int comm_val(int cid,int rank,int t,int i){
  return (int)(((unsigned)(cid*2654435761u) ^ (unsigned)(rank*2246822519u)
              ^ (unsigned)(t*40503u) ^ (unsigned)(i*668265263u)) & 0x7fffffff);
}
__host__ __device__ __forceinline__ int priv_val(int blk,int i){
  return (int)(((unsigned)(blk*2654435761u) ^ (unsigned)(i*374761393u)
              ^ 0x9e3779b9u) & 0x0000ffff);
}
// task type of item t: 1 == LARGE (A), 0 == SMALL communicating (B).
// INTERLEAVED so ONE resident CTA sees BOTH over its lifetime.
__host__ __device__ __forceinline__ int is_big(int t,int big_every){
  if (big_every <= 0) return 0;
  if (big_every == 1) return 1;
  return (t % big_every) == 0;
}
// PAYLOAD SIZE of a task item, in ints.  This is the size that decides
// LEGALITY: a type's payload is proportional to the pages ITS OWN type needs, so
// the LARGE type's payload is (ARENA_PAGES/SMALL_PAGES)x the small one and
// therefore CANNOT fit an arena sized for the small type.  That is what makes
// ARM 4 (bucketed) illegal under a heterogeneous mix, and it is arithmetic, not
// an assertion.
//   small payload = comm_ints
//   big payload   = comm_ints * ARENA_PAGES / SMALL_PAGES
__host__ __device__ __forceinline__ int task_ints(int t,int big_every,
                                                  int comm_ints){
  return is_big(t,big_every)
       ? (comm_ints * ARENA_PAGES) / SMALL_PAGES
       : comm_ints;
}
// Bounded slice actually TOUCHED per task (the disclosed modelling convenience).
// The full payload above is what is RESERVED and what decides legality; only
// this many ints are read/written, so simulated cycles do not scale with arena
// bytes.  TOUCH_CAP applies identically in every arm.
__host__ __device__ __forceinline__ int touch_ints(int n){
  return (n < TOUCH_CAP) ? n : TOUCH_CAP;
}

extern __shared__ int arena[];   // dynamic: a launch scalar, never DCE-able

template<int CS>
__global__ void __cluster_dims__(CS,1,1) __launch_bounds__(64,1)
maxunion_kernel(int* __restrict__ g_comm, const int* __restrict__ g_priv,
                int num_clusters, int ntasks, int big_every,
                int comm_lines, int priv_lines){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int blk  = (int)blockIdx.x;
  const int cid  = blk / CS;
  const int comm_ints = comm_lines * INTS_PER_LINE;
  const int priv_ints = priv_lines * INTS_PER_LINE;
  // widest payload any item communicates -- the global arena must hold it
  const int max_ints  = (comm_ints * ARENA_PAGES) / SMALL_PAGES;
  const int arena_ints = (RESERVED_PAGES * PAGE_BYTES) / 4;
  size_t rank_base = (((size_t)cid*CS + rank)*(size_t)ntasks) * (size_t)max_ints;
  const int* mypriv = g_priv + (size_t)blk * (size_t)priv_ints;

#if ARM == 2 || ARM == 3
  // MARKED: RESC registers its own comm range (distributed producers), exactly
  // as markfrac.cu / pipe_chain.cu do.
  if (tid < 32)
    register_range_warp0(g_comm + rank_base,
                         (size_t)ntasks*max_ints*sizeof(int));
#endif
  __syncthreads();
  cl.sync();

  long long acc_u = 0, acc_m = 0;

  for (int t = 0; t < ntasks; ++t) {
    const int n  = task_ints(t, big_every, comm_ints);   // RESERVED size (legality)
    const int nt = touch_ints(n);                        // TOUCHED size (cycles)

    // ---- PRIVATE working set: re-read once per task.  This is the CO-RESIDENT
    // COMPETITOR for L1 ways -- the thing markfrac at f=1.0 does not provide and
    // the reason the retention question is even askable here.  Read with
    // ld.global.ca in EVERY arm, so no arm is charged a cost the others avoid.
    for (int i = tid; i < priv_ints; i += blockDim.x)
      acc_u += ld_ca_s32(&mypriv[i]);

    // ---- PRODUCE this item's payload to global (all arms, identical traffic).
    int* mine = g_comm + rank_base + (size_t)t*max_ints;
    for (int i = tid; i < nt; i += blockDim.x)
      st_wb_s32(&mine[i], comm_val(cid, rank, t, i));

#if STAGES_IN_SMEM
    // ---- INCUMBENT (DSM): stage the payload in the SMEM arena, peers read it
    // via map_shared_rank.  ARM 1 rotates the slot inside the SAME fixed arena
    // (reallocation, not shrinkage).  ARM 4's arena is sized for the SMALL type,
    // so a LARGE payload DOES NOT FIT -- which is exactly why bucketing is
    // illegal under a heterogeneous mix, and it shows up as a checksum mismatch
    // rather than as prose.  LEGALITY is tested against the RESERVED size n,
    // not the touched size, because the reservation is what software must make.
    //
    // ARM 1 slot arithmetic, FIXED (two bugs found by A1 reporting
    // BITIDENTICAL=0 -- both mine, neither the mechanism's):
    //  (1) a naive `slot = (t % ARENA_PAGES) * PAGE` left cap = arena - slot,
    //      which is SMALLER than the big payload for t%13 in {10,11,12}, so the
    //      producer staged nothing and those payloads were silently LOST.  A
    //      recycling policy that cannot hold the item it is handed is not
    //      "recycling", it is a bug: real page recycling reuses whole pages and
    //      always leaves room for the largest item (Stanford's ret_order[13]
    //      permutes all 13 pages, it does not shrink the usable arena).  The
    //      slot is therefore rotated MODULO THE HEADROOM, so every task still
    //      gets the full payload's worth of space.
    //  (2) the consumer used slot(t) -- its OWN task index -- but the data it
    //      wants was staged by the peer at slot(t-1).  With a rotating slot
    //      those differ, so ARM 1 read the WRONG offset even when capacity
    //      allowed.  The consumer now computes the PRODUCER's slot, slot(t-1).
    // ARM 0 / ARM 4 use slot = 0 always and were never affected (A0 measured
    // BITIDENTICAL=1).
#if ARM == 1
    // headroom in whole pages that still fits the widest payload
    const int max_pages_needed = (max_ints * 4 + PAGE_BYTES - 1) / PAGE_BYTES;
    const int rot_pages = (ARENA_PAGES - max_pages_needed) + 1;   // >= 1
    const int slot      = (((t     % rot_pages) * PAGE_BYTES) / 4);
    const int slot_prod = ((((t-1) % rot_pages) * PAGE_BYTES) / 4);
#else
    const int slot      = 0;
    const int slot_prod = 0;
#endif
    const int cap = (arena_ints > slot) ? (arena_ints - slot) : 0;
    const int fits = (n <= cap);            // does the WHOLE payload fit?
    const int nst  = fits ? nt : 0;         // if not, this arm cannot stage it
    for (int i = tid; i < nst; i += blockDim.x)
      arena[slot + i] = comm_val(cid, rank, t, i);
    __syncthreads();
    cl.sync();
    if (rank > 0 && t >= 1) {
      const int np    = task_ints(t-1, big_every, comm_ints);
      const int npt   = touch_ints(np);
      // capacity available at the PRODUCER's slot, not the consumer's
      const int pcap  = (arena_ints > slot_prod) ? (arena_ints - slot_prod) : 0;
      const int pfits = (np <= pcap);
      // Peer rank, forced into range by MASKING rather than by a ternary.
      // WHY: nvcc hoists the mapa address computation into the loop preheader,
      // so rank 0 evaluates it even though the guarded body never runs -- and a
      // ternary clamp is optimised straight back to `add.s32 %r38, %r1, -1`
      // (verified in the emitted PTX), giving rank 0xFFFFFFFF and
      // "Mapa Error! Target CTA Rank: 4294967295".  A mask cannot be folded
      // away: for rank >= 1 it is exactly rank-1, and for rank 0 it yields a
      // valid in-cluster rank whose mapped address is never dereferenced.
      const unsigned peer_rank = ((unsigned)(rank - 1)) & (unsigned)(CS - 1);
      int* peer = (int*)cl.map_shared_rank(arena, peer_rank);
      // A payload the producer could not stage is NOT readable: the consumer
      // gets nothing, the checksum falls short of the reference, and ARM 4
      // reports BITIDENTICAL=0.  That is the measured illegality of bucketing.
      if (pfits)
        for (int i = tid; i < npt; i += blockDim.x)
          acc_m += peer[slot_prod + i];   // the PRODUCER's slot, not our own
    }
#else
    // ---- RESC: no SMEM staging.  The neighbour's payload is read from GLOBAL
    // with .ca and the line is MARKED, so the cluster read-share can serve it
    // from the producer's L1 instead of L2.  RESC reserves nothing, so no
    // payload can fail to fit -- it degrades to L2, it never becomes illegal.
    cl.sync();
    if (rank > 0 && t >= 1) {
      const int npt = touch_ints(task_ints(t-1, big_every, comm_ints));
      size_t nbr_base = (((size_t)cid*CS + (rank-1))*(size_t)ntasks)
                        * (size_t)max_ints;
      const int* nbr = g_comm + nbr_base + (size_t)(t-1)*max_ints;
      for (int i = tid; i < npt; i += blockDim.x)
        acc_m += ld_ca_s32(&nbr[i]);
    }
#endif
    cl.sync();   // task boundary / epoch tick
  }
  long long acc = acc_u + acc_m;
  if (acc) atomicAdd(&g_checksum, (unsigned long long)acc);
#endif
}

int main(int argc, char** argv){
  int num_clusters = (argc>1)? atoi(argv[1]) : 2;
  int ntasks       = (argc>2)? atoi(argv[2]) : 32;
  int big_every    = (argc>3)? atoi(argv[3]) : 4;
  int comm_lines   = (argc>4)? atoi(argv[4]) : 4;
  int priv_lines   = (argc>5)? atoi(argv[5]) : 256;
  if (num_clusters<1) num_clusters=1;
  if (ntasks<1) ntasks=1; if (ntasks>MAX_TASKS) ntasks=MAX_TASKS;
  if (big_every<0) big_every=0;
  if (comm_lines<1) comm_lines=1; if (priv_lines<0) priv_lines=0;
  const int CS=CLUSTER_SIZE;
  const int comm_ints = comm_lines*INTS_PER_LINE;
  const int priv_ints = priv_lines*INTS_PER_LINE;
  const int max_ints  = (comm_ints*ARENA_PAGES)/SMALL_PAGES;
  const int nblocks   = CS*num_clusters;
  const size_t dyn_smem = (size_t)RESERVED_PAGES * PAGE_BYTES;

  size_t ncomm=(size_t)num_clusters*CS*(size_t)ntasks*max_ints;
  size_t npriv=(size_t)nblocks*(size_t)priv_ints;
  int *d_comm=nullptr, *d_priv=nullptr;
  CUDA_CHECK(cudaMalloc(&d_comm, ncomm*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_comm, 0, ncomm*sizeof(int)));
  if (npriv){
    int* h_priv=(int*)malloc(npriv*sizeof(int));
    for (int b=0;b<nblocks;++b)
      for (int i=0;i<priv_ints;++i) h_priv[(size_t)b*priv_ints+i]=priv_val(b,i);
    CUDA_CHECK(cudaMalloc(&d_priv, npriv*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_priv,h_priv,npriv*sizeof(int),cudaMemcpyHostToDevice));
    free(h_priv);
  } else {
    CUDA_CHECK(cudaMalloc(&d_priv, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_priv, 0, sizeof(int)));
  }

  // >48 KB per block requires dynamic SMEM plus this opt-in (CUDA C++
  // Programming Guide: "Kernels relying on shared memory allocations over 48 KB
  // per block are architecture-specific ... must use dynamic shared memory").
  if (dyn_smem > 0)
    CUDA_CHECK(cudaFuncSetAttribute(maxunion_kernel<CLUSTER_SIZE>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_smem));

  dim3 grid(nblocks,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  maxunion_kernel<CLUSTER_SIZE><<<grid,block,dyn_smem>>>(
      d_comm,d_priv,num_clusters,ntasks,big_every,comm_lines,priv_lines);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long dev=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&dev, g_checksum, sizeof(dev)));

  // Reference: the FULL, correct answer -- every consumer sees its producer's
  // whole TOUCHED payload.  An arm whose arena cannot RESERVE that payload
  // (ARM 4 with big tasks) stages nothing for it, so its consumer reads nothing
  // and it legitimately fails this check.  That failure is the point.
  unsigned long long ref=0;
  {
    long long ps=0;
    for (int b=0;b<nblocks;++b){
      long long s=0;
      for (int i=0;i<priv_ints;++i) s+=priv_val(b,i);
      ps += s*(long long)ntasks;
    }
    long long cs=0;
    for (int cid=0; cid<num_clusters; ++cid)
      for (int r=1;r<CS;++r)
        for (int t=1;t<ntasks;++t){
          const int npt = touch_ints(task_ints(t-1,big_every,comm_ints));
          long long s=0;
          for (int i=0;i<npt;++i) s+=comm_val(cid,r-1,t-1,i);
          cs+=s;
        }
    ref=(unsigned long long)(ps+cs);
  }
  int nbig=0; for(int t=0;t<ntasks;++t) nbig+=is_big(t,big_every);
  const int reserved_B  = (int)dyn_smem;
  const int worst_B     = ARENA_PAGES * PAGE_BYTES;
  const int small_B     = SMALL_PAGES * PAGE_BYTES;
  // payload sizes, in bytes, that decide legality
  const int small_payload_B = comm_ints*4;
  const int big_payload_B   = max_ints*4;
  // does THIS arm's arena hold the LARGEST payload it may be handed?
  const int holds_big = (!STAGES_IN_SMEM) || (max_ints*4 <= reserved_B);
  // The max-union tax itself: reserved-but-idle while a SMALL item runs.
  const double idle_frac = (reserved_B > 0)
      ? (double)(reserved_B - small_B) / (double)reserved_B : 0.0;
  // G12 (ADDENDUM 03): the TOUCHED work per task type and their ratio.  If this
  // ratio is 1.00 the heterogeneity axis is inert BY CONSTRUCTION and no null on
  // the big_every axis may be reported from this point.
  const int touched_small = touch_ints(comm_ints);
  const int touched_big   = touch_ints(max_ints);
  const double touched_ratio = (touched_small > 0)
      ? (double)touched_big / (double)touched_small : 0.0;
  printf("MAXUNION arm=%d cluster=%d num_clusters=%d ntasks=%d big_every=%d "
         "nbig=%d nsmall=%d comm_lines=%d priv_lines=%d comm_B=%d priv_B=%d "
         "touch_cap=%d small_payload_B=%d big_payload_B=%d "
         "touched_small_ints=%d touched_big_ints=%d touched_ratio=%.4f "
         "arena_pages_reserved=%d arena_pages_worstcase=%d small_pages=%d "
         "page_B=%d SMEM_PER_CTA_B=%d worstcase_arena_B=%d small_arena_B=%d "
         "idle_reserved_frac=%.6f stages_in_smem=%d holds_big_payload=%d "
         "OUT_CHECKSUM=%llu REF_CHECKSUM=%llu BITIDENTICAL=%d\n",
         ARM, CS, num_clusters, ntasks, big_every, nbig, ntasks-nbig,
         comm_lines, priv_lines, comm_lines*LINE_BYTES, priv_lines*LINE_BYTES,
         TOUCH_CAP, small_payload_B, big_payload_B,
         touched_small, touched_big, touched_ratio,
         RESERVED_PAGES, ARENA_PAGES, SMALL_PAGES, PAGE_BYTES,
         reserved_B, worst_B, small_B, idle_frac, (int)(STAGES_IN_SMEM),
         holds_big, dev, ref, (dev==ref)?1:0);
  CUDA_CHECK(cudaFree(d_comm)); CUDA_CHECK(cudaFree(d_priv));
  return 0;
}
