// markfrac.cu -- MARKED-FRACTION SWEEP.
//
// Purpose.  The selective-invalidate experiment
// (selective-invalidate-methodology.md) showed that RESC's marked-line
// selective invalidate demonstrably WORKS (positive control selprobe.cu:
// unmarked L1 hit rate 0.967628 under knob 2, bit-identical to knob 0, while
// the marked buffer is still invalidated) but is worth EXACTLY NOTHING on the
// paper's own ablation workloads, because pipe_chain and resc_e2e register
// essentially ALL of their read traffic as marked
// (unmarked_rd_hit = unmarked_rd_miss = 0 in all 64 pipeline and all 240
// legality runs).  They sit at marked fraction = 1.0.
//
// This benchmark makes the marked fraction a SWEPT PARAMETER instead of an
// accident of benchmark construction, so the paper can report WHERE the effect
// lives and where it vanishes rather than one hand-picked point.
//
// Construction.  This is pipe_chain PLUS a per-block private read-only tile.
// The communication pattern, its size, its lag and its arena layout are
// byte-for-byte pipe_chain's (pipe_chain.cu:73-105): each rank produces its own
// comm tile every epoch and consumes its LEFT neighbour's tile from LAG epochs
// earlier, fan-in 1.  The ONLY thing added is PRIV_LINES lines of per-block
// PRIVATE, READ-ONLY, NEVER-REGISTERED data that the block re-reads once per
// epoch.  Therefore:
//
//     marked fraction f = comm_lines / (comm_lines + priv_lines)
//
//   * priv_lines = 0  =>  f = 1.0  =>  THIS KERNEL IS pipe_chain.
//     That is the self-consistency anchor: it must reproduce pipe_chain's
//     behaviour (knob 2 == knob 1) at that end of the sweep.
//   * priv_lines large => f -> 0 => a large reused unmarked working set is
//     co-resident with the same small marked communication buffer.
//
// COMMUNICATION IS HELD FIXED ACROSS THE ENTIRE SWEEP (same comm_lines, same
// lag, same producers, same consumers, same bytes).  Only the private tile
// changes.  Any movement across the sweep is therefore attributable to the
// private (unmarked) working set and nothing else.
//
// The private tile is READ-ONLY and PER-BLOCK PRIVATE.  This is what makes
// preserving it across a cluster acquire SOUND rather than an optimisation
// cheat: a line no peer can have written cannot be stale.  It is the exact
// property the pre-registration named as the correctness argument for knob 2
// (selective-invalidate-prereg.md S0a).
//
// ANTI-RIGGING NOTE (read this before believing any win).  The private tile is
// re-read ONCE PER EPOCH and is never written by anyone.  A real programmer
// facing a whole-L1 invalidate could in principle hoist it into registers or
// SMEM if it were small enough, or use a non-coherent/read-only path.  Two
// things are done about that rather than hidden:
//   (1) the private tile is read with ld.global.ca in EVERY arm, so no arm is
//       charged a cost the others avoid;
//   (2) the marked fraction at which any win appears is REPORTED, so a reader
//       can judge whether real kernels operate there (KILL-W3).
//
// ARMS.  Selected at COMPILE TIME by -DARM=n.  The private tile is read with
// ld.global.ca in EVERY arm -- no arm is charged a cost real software can
// avoid on data that is private and read-only.  The arms differ ONLY in how the
// COMMUNICATED data is handled:
//
//   ARM=0  "cg"   correct always-global incumbent.  Comm reads use
//                 ld.global.cg (L1-bypassing), so no comm line can be stale in
//                 L1.  Not registered.  No RESC hardware.
//   ARM=1  "ca"   unsafe overlay.  Comm reads use ld.global.ca, so a stale L1
//                 copy CAN be read.  Not registered.  No RESC hardware.  Used
//                 as a deliberately CONSERVATIVE stand-in for a legal
//                 DSM/SMEM-staging arm: it pays no staging cost at all, so it
//                 is strictly faster than a real DSM arm would be
//                 (selective-invalidate-prereg.md S3).
//   ARM=2  "resc" comm reads use ld.global.ca AND the comm tile is REGISTERED
//                 with gpgpusim_register_cluster_coherent_addr, i.e. the comm
//                 lines are MARKED and the private tile is not.  arm2 vs arm3
//                 is then chosen at RUN TIME by RESC_GEN_RING / RESC_MW exactly
//                 as in pipe_chain, so arm2 and arm3 are the SAME BINARY.
//
// Args: <num_clusters> <nstages> <lag> <comm_lines> <priv_lines> [reuse]
//
// REUSE VARIANT (2026-09-04): identical to markfrac.cu except the consumer re-reads
// the communicated tile `reuse` times per epoch.  reuse=1 is markfrac.cu exactly.
// This is the CROSSED CELL: private tile present (f=0.1111) AND reuse >= 1, which
// neither the legality axis (reuse swept, zero private tile) nor the markfrac sweep
// (marked fraction swept, reuse=1) ever measured.
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
#ifndef MAX_STAGES
#define MAX_STAGES 512
#endif
#define LINE_BYTES 128
#define INTS_PER_LINE (LINE_BYTES/4)

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// simulator hook: intercepted in cuda-sim/instructions.cc:2761
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
__device__ __forceinline__ int ld_cg_s32(const int* p){ int v;
  asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_wb_s32(int* p,int v){
  asm volatile("st.global.wb.s32 [%0],%1;" :: "l"(p),"r"(v) : "memory"); }

// Comm-read policy = the ONLY per-arm difference.
__device__ __forceinline__ int ld_comm_s32(const int* p){
#if ARM == 0
  return ld_cg_s32(p);     // correct always-global incumbent
#else
  return ld_ca_s32(p);     // ca overlay (ARM 1) and RESC (ARM 2)
#endif
}

__device__ unsigned long long g_checksum = 0;

// deterministic value generators, shared host/device so the reference is exact
__host__ __device__ __forceinline__ int comm_val(int cid,int rank,int gen,int i){
  return (int)(((unsigned)(cid*2654435761u) ^ (unsigned)(rank*2246822519u)
              ^ (unsigned)(gen*40503u) ^ (unsigned)(i*668265263u)) & 0x7fffffff);
}
__host__ __device__ __forceinline__ int priv_val(int blk,int i){
  return (int)(((unsigned)(blk*2654435761u) ^ (unsigned)(i*374761393u)
              ^ 0x9e3779b9u) & 0x0000ffff);
}

// comm arena:  g_comm[ ((cid*CS + rank)*nstages + gen)*comm_ints + i ]
// priv arena:  g_priv[ blk*priv_ints + i ]                (read-only, unmarked)
template<int CS>
__global__ void __cluster_dims__(CS,1,1)
markfrac_kernel(int* __restrict__ g_comm, const int* __restrict__ g_priv,
                int num_clusters, int nstages, int lag,
                int comm_lines, int priv_lines, int reuse){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int blk  = (int)blockIdx.x;
  const int cid  = blk / CS;
  const int comm_ints = comm_lines * INTS_PER_LINE;
  const int priv_ints = priv_lines * INTS_PER_LINE;
  size_t rank_base = (((size_t)cid*CS + rank)*nstages) * (size_t)comm_ints;
  const int* mypriv = g_priv + (size_t)blk * (size_t)priv_ints;

  // MARKED: each rank registers ITS OWN comm tile range (distributed
  // producers), exactly as pipe_chain.cu:87 does.  RESC arm only.
#if ARM == 2
  if (tid < 32)
    register_range_warp0(g_comm + rank_base,
                         (size_t)nstages*comm_ints*sizeof(int));
#endif
  __syncthreads();
  cl.sync();

  // Epoch body below is pipe_chain.cu:96-105 VERBATIM in structure -- produce,
  // consume, cl.sync() at the END -- with exactly one thing inserted: the
  // private read at the top.  At priv_lines == 0 that insertion is a zero-trip
  // loop, so THIS KERNEL IS pipe_chain at the f = 1.0 end of the sweep.
  long long acc_u = 0, acc_m = 0;
  for (int e = 0; e < nstages; ++e) {
    // ---- UNMARKED: re-read the private read-only tile, once per epoch.
    // A whole-L1 acquire invalidate destroys this every epoch; a marked-line
    // selective invalidate must leave it alone.  Read with .ca in EVERY arm.
    for (int i = tid; i < priv_ints; i += blockDim.x)
      acc_u += ld_ca_s32(&mypriv[i]);

    // ---- MARKED: produce my own generation-e comm tile.
    int* mine = g_comm + rank_base + (size_t)e*comm_ints;
    for (int i = tid; i < comm_ints; i += blockDim.x)
      st_wb_s32(&mine[i], comm_val(cid, rank, e, i));

    // ---- MARKED: consume LEFT neighbour's tile from epoch (e-lag), REUSE times.
    // The REUSE loop is the ONLY addition vs markfrac.cu.  At reuse == 1 it is a
    // single-trip loop and this kernel is markfrac.cu exactly -- that identity is
    // what makes the R=1 abort-check in the pre-registration meaningful.
    // Each extra pass re-reads the SAME communicated lines: for a baseline that
    // bypasses L1 (ld.cg) every pass is another L2 round trip, while a cache that
    // holds the line serves passes 2..R as hits.  This is the amortization axis.
    if (rank > 0 && e >= lag) {
      int g = e - lag;
      size_t nbr_base = (((size_t)cid*CS + (rank-1))*nstages) * (size_t)comm_ints;
      const int* nbr = g_comm + nbr_base + (size_t)g*comm_ints;
      for (int r = 0; r < reuse; ++r)
        for (int i = tid; i < comm_ints; i += blockDim.x)
          acc_m += ld_comm_s32(&nbr[i]);
    }
    cl.sync();  // epoch tick / THE ACQUIRE under test
  }
  long long acc = acc_u + acc_m;
  if (acc) atomicAdd(&g_checksum, (unsigned long long)acc);
#endif
}

int main(int argc, char** argv){
  int num_clusters = (argc>1)? atoi(argv[1]) : 2;
  int nstages      = (argc>2)? atoi(argv[2]) : 32;
  int lag          = (argc>3)? atoi(argv[3]) : 2;
  int comm_lines   = (argc>4)? atoi(argv[4]) : 4;
  int priv_lines   = (argc>5)? atoi(argv[5]) : 0;
  int reuse        = (argc>6)? atoi(argv[6]) : 1;   // consumer passes per epoch
  if (num_clusters<1) num_clusters=1;
  if (nstages<1) nstages=1; if (nstages>MAX_STAGES) nstages=MAX_STAGES;
  if (lag<0) lag=0; if (comm_lines<1) comm_lines=1; if (priv_lines<0) priv_lines=0;
  if (reuse<1) reuse=1;
  const int CS=CLUSTER_SIZE;
  const int comm_ints = comm_lines*INTS_PER_LINE;
  const int priv_ints = priv_lines*INTS_PER_LINE;
  const int nblocks   = CS*num_clusters;

  size_t ncomm=(size_t)num_clusters*CS*nstages*comm_ints;
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

  dim3 grid(nblocks,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  markfrac_kernel<CLUSTER_SIZE><<<grid,block>>>(d_comm,d_priv,num_clusters,
                                                nstages,lag,comm_lines,priv_lines,reuse);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long dev=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&dev, g_checksum, sizeof(dev)));

  // reference: private part = every block re-reads its whole tile every epoch;
  //            comm part    = rank r (1..CS-1) reads rank(r-1) gen(e-lag).
  unsigned long long ref=0;
  {
    long long ps=0;
    for (int b=0;b<nblocks;++b){
      long long s=0;
      for (int i=0;i<priv_ints;++i) s+=priv_val(b,i);
      ps += s*(long long)nstages;
    }
    long long cs=0;
    for (int cid=0; cid<num_clusters; ++cid)
      for (int r=1;r<CS;++r)
        for (int e=lag;e<nstages;++e){
          int g=e-lag; long long s=0;
          for (int i=0;i<comm_ints;++i) s+=comm_val(cid,r-1,g,i);
          cs += s*(long long)reuse;   // consumer reads the tile `reuse` times
        }
    ref=(unsigned long long)(ps+cs);
  }
  const double f = (double)comm_lines/(double)(comm_lines+priv_lines);
  printf("MARKFRAC arm=%d cluster=%d num_clusters=%d nstages=%d lag=%d "
         "comm_lines=%d priv_lines=%d comm_B=%d priv_B=%d markfrac=%.6f reuse=%d "
         "OUT_CHECKSUM=%llu REF_CHECKSUM=%llu BITIDENTICAL=%d\n",
         ARM, CS, num_clusters, nstages, lag, comm_lines, priv_lines,
         comm_lines*LINE_BYTES, priv_lines*LINE_BYTES, f, reuse,
         dev, ref, (dev==ref)?1:0);
  CUDA_CHECK(cudaFree(d_comm)); CUDA_CHECK(cudaFree(d_priv));
  return 0;
}
