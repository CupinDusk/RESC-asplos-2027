// RESC MULTI-WRITER -- PRIMARY workload: cross-block reduction into a small
// cluster-resident accumulator.
//
// Two conflict forms (select with -DFORM):
//   FORM=0  ATOMIC : every block RMWs ONE accumulator word acc0 with atomicAdd
//                    (all C blocks hammer one word -> stresses path B, the
//                    owner-token hand-off; makes FORWARD/invalidate live).
//   FORM=1  FALSE-SHARING MARKED-STORE : each rank writes its OWN word `rank`
//                    of a shared 128B line each round (distinct words, same line
//                    = false sharing -> stresses path A, the sync-merge). rank0
//                    then READS all words to combine (exercises the MW_MERGE
//                    merge-read serve path).
//
// INTEGER ONLY so the CPU reference is bit-exact. All accumulators are int64.
//
// Arms are selected sim-side by env RESC_MW (0=arm3 old-escape, 1=arm2 MW) and
// RESC_MW_LWW=1 (arm4 whole-line-LWW ablation). Arm1 (pure-global baseline) is a
// separate build with -DRESC_MW_BASELINE_BUILD (no register hook -> everything
// goes to global/L2, no RESC state at all).
//
// Layout per cluster: WORDS_PER_CLUSTER int64, contiguous; 8 int64 = 64B, so a
// cluster's words share ONE 128B line (false sharing is real).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8            // C blocks per cluster (portable <=8)
#endif
#ifndef FORM
#define FORM 0                    // 0=atomic (path B), 1=false-sharing store (path A)
#endif

// One 128B line per cluster (16 int64 max); we use CLUSTER_SIZE (+1 combine) words.
#define WORDS_PER_CLUSTER 16

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// Sim hook: mark a global address as a cluster-coherent RESC communication line.
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  asm volatile("" :: "l"(addr) : "memory");   // intercepted functionally by sim
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)128 - 1);
}
__device__ __forceinline__ void register_range_warp0(void* base, size_t bytes){
  uintptr_t b = (uintptr_t)base;
  uintptr_t start = align_down_line(b);
  int nlines = (int)(((b + bytes - start) + 127) / 128);
  int lane = threadIdx.x & 31;
  for(int off=0; off<nlines; off+=32){
    int idx = off + lane;
    uintptr_t line = start + (uintptr_t)((idx < nlines) ? idx : 0) * 128;
    gpgpusim_register_cluster_coherent_addr((void*)line);
  }
}

// Deterministic per-(cluster,rank,round) integer contribution -> bit-exact ref.
__host__ __device__ __forceinline__ long long contrib(int cid, int rank, int r){
  return (long long)( (cid*131 + rank*17 + r*7 + 1) & 0xFFFF );
}

// ld/st marked comm helpers (write-back / cache-all so RESC state is exercised).
__device__ __forceinline__ long long ld_ca_s64(const long long* p){ long long v;
  asm volatile("ld.global.ca.s64 %0,[%1];" : "=l"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_wb_s64(long long* p, long long v){
  asm volatile("st.global.wb.s64 [%0],%1;" :: "l"(p),"l"(v) : "memory"); }

template<int CS>
__global__ void __cluster_dims__(CS,1,1)
reduce_kernel(long long* __restrict__ acc, int num_clusters, int rounds){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  long long* my_acc = acc + (size_t)cid * WORDS_PER_CLUSTER;

#if !defined(RESC_MW_BASELINE_BUILD)
  if (rank == 0 && tid < 32) {
    register_range_warp0(my_acc, (size_t)WORDS_PER_CLUSTER * sizeof(long long));
  }
#endif
  cl.sync();

#if FORM == 0
  // ATOMIC: all C blocks RMW the SAME 32-bit word acc32[0] every round (path B
  // torture). 32-bit atomics match the sim's supported width (int64 atomics hit a
  // pre-existing 8-byte byte-mask overflow in the coalescer, unrelated to RESC).
  // We CONSUME the atomicAdd return value (accumulate into acc32[1]) so the
  // compiler emits a true `atom.global.add` (a read-modify-write that returns the
  // prior value) rather than a write-only `red.global` -- the latter is treated
  // as a store and would never traverse the atomic-token read path.
  int* acc32 = (int*)my_acc;                 // reuse the same line region
  long long prior_sum = 0;
  for (int r = 0; r < rounds; ++r) {
    if (tid == 0) {
      int prev = atomicAdd(&acc32[0], (int)contrib(cid, rank, r));
      prior_sum += prev;                     // consume return -> forces atom (not red)
    }
    cl.sync();
  }
  if (tid == 0) acc32[2] = (int)prior_sum;   // keep it live (not required for ref)
#else
  // FALSE-SHARING: each rank stores its running sum into its OWN word `rank` of
  // the shared line each round (distinct words, same 128B line = false sharing;
  // path A). To make the line genuinely cluster-resident (so a 2nd writer finds
  // a peer copy), every rank first READS a neighbor's word each round (this also
  // exercises the MW_MERGE merge-read serve path once merged). rank0 combines by
  // READING all words at the end.
  long long racc = 0;
  for (int r = 0; r < rounds; ++r) {
    if (tid == 0) {
      // read a neighbor's word (makes the line resident in this L1 as reader,
      // and once co-written triggers the MW_MERGE merge-read invariant).
      volatile long long peekv = ld_ca_s64(&my_acc[(rank + 1) % CS]);
      (void)peekv;
      racc += contrib(cid, rank, r);
      st_wb_s64(&my_acc[rank], racc);   // marked store to distinct word of line
    }
    cl.sync();
  }
  if (rank == 0 && tid == 0) {
    long long s = 0;
    for (int k = 0; k < CS; ++k) s += ld_ca_s64(&my_acc[k]);  // merge-read
    st_wb_s64(&my_acc[WORDS_PER_CLUSTER-1], s);               // combined result
  }
  cl.sync();
#endif
#endif
}

int main(int argc, char** argv){
  int num_clusters = (argc>1)? atoi(argv[1]) : 64;
  int rounds       = (argc>2)? atoi(argv[2]) : 50;
  if (num_clusters<1) num_clusters=1;
  if (rounds<1) rounds=1;

  const int CS = CLUSTER_SIZE;
  const size_t nwords = (size_t)num_clusters * WORDS_PER_CLUSTER;
  long long* d_acc=nullptr;
  CUDA_CHECK(cudaMalloc(&d_acc, nwords*sizeof(long long)));
  CUDA_CHECK(cudaMemset(d_acc, 0, nwords*sizeof(long long)));

  dim3 grid(CS*num_clusters,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  reduce_kernel<CLUSTER_SIZE><<<grid,block>>>(d_acc, num_clusters, rounds);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  long long* h_acc = (long long*)malloc(nwords*sizeof(long long));
  CUDA_CHECK(cudaMemcpy(h_acc, d_acc, nwords*sizeof(long long), cudaMemcpyDeviceToHost));

  // ---- bit-exact CPU reference ----
#if FORM == 0
  // Device wrote an int32 sum into the low word of each cluster's line region.
  // Compare as int32 to avoid int64-alias high-bit noise.
  int* h_acc32 = (int*)h_acc;
  long long dev_checksum=0, ref_checksum=0; int mism=0, first_mis=-1;
  for (int cid=0; cid<num_clusters; ++cid){
    long long s=0;
    for (int r=0;r<rounds;++r) for (int rank=0; rank<CS; ++rank) s += contrib(cid,rank,r);
    int refv = (int)s;
    int devv = h_acc32[(size_t)cid*WORDS_PER_CLUSTER*2 + 0];  // int32 stride within int64 words
    dev_checksum += devv; ref_checksum += refv;
    if (devv!=refv){ mism++; if(first_mis<0) first_mis=cid; }
  }
  printf("MW_REDUCE FORM=%d cluster=%d num_clusters=%d rounds=%d "
         "OUT_CHECKSUM=%lld REF_CHECKSUM=%lld BITIDENTICAL=%d mismatches=%d first_mis=%d\n",
         FORM, CS, num_clusters, rounds,
         dev_checksum, ref_checksum, (dev_checksum==ref_checksum && mism==0)?1:0,
         mism, first_mis);
  free(h_acc);
  CUDA_CHECK(cudaFree(d_acc));
  return 0;
#else
  long long* ref = (long long*)calloc(nwords, sizeof(long long));
  for (int cid=0; cid<num_clusters; ++cid){
    long long combined=0;
    for (int rank=0; rank<CS; ++rank){
      long long racc=0;
      for (int r=0;r<rounds;++r) racc += contrib(cid,rank,r);
      ref[(size_t)cid*WORDS_PER_CLUSTER + rank] = racc;   // each word = rank's sum
      combined += racc;
    }
    ref[(size_t)cid*WORDS_PER_CLUSTER + (WORDS_PER_CLUSTER-1)] = combined;
  }

  long long dev_checksum=0, ref_checksum=0; int mism=0, first_mis=-1;
  for (size_t i=0;i<nwords;++i){
    dev_checksum += h_acc[i]; ref_checksum += ref[i];
    if (h_acc[i]!=ref[i]){ mism++; if(first_mis<0) first_mis=(int)i; }
  }

  printf("MW_REDUCE FORM=%d cluster=%d num_clusters=%d rounds=%d "
         "OUT_CHECKSUM=%lld REF_CHECKSUM=%lld BITIDENTICAL=%d mismatches=%d first_mis=%d\n",
         FORM, CS, num_clusters, rounds,
         dev_checksum, ref_checksum, (dev_checksum==ref_checksum && mism==0)?1:0,
         mism, first_mis);

  free(ref); free(h_acc);
  CUDA_CHECK(cudaFree(d_acc));
  return 0;
#endif
}
