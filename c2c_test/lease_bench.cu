// RESC EPOCH/VERSION LEASE cross-epoch-reuse benchmark (candidate #1).
//
// Regime the lease targets (pre-reg): a producer writes a marked communication
// TABLE that same-cluster consumers RE-READ across several cluster.sync epochs
// before (part of) it is updated. This is a CONSUMER-side read that arm2 would
// needlessly re-fetch every epoch, NOT a producer-side lagged forward (=RING).
//
// Cluster (CGA) of CS CTAs: rank 0 = PRODUCER of a TILE-line shared table;
// ranks 1..CS-1 = CONSUMERS that sum ALL table lines every epoch. Each epoch e
// the producer REWRITES a fraction f of the lines (bumping their version); the
// rest keep their value+version. Consumers re-read across cluster.sync.
//
// Same binary for every arm; the coherence discipline is chosen ONLY by the
// sim env (RESC_LEASE / RESC_LEASE_S / RESC_LEASE_NOKEEP), exactly like the
// existing pipe_chain uses RESC_GEN_RING. Loads are plain cacheable (ld.ca) so
// L1 residency across epochs is POSSIBLE; whether a re-read hits L1 (kept) or
// re-fetches (drained) is decided sim-side per arm.
//   RESC_LEASE=0 : off (baseline; marked lines persist -> perfect keep)
//   RESC_LEASE=1 : arm0 always-global (every marked read re-fetches)
//   RESC_LEASE=2 : arm2 DeNovo/hLRC (self-invalidate ALL marked lines each sync)
//   RESC_LEASE=3 : arm-lease (keep across epoch iff lease live AND version dominates)
//   RESC_LEASE_S=<S> : lease budget in epochs (S=1 => arm-lease == arm2)
//   RESC_LEASE_NOKEEP=1 : disable-lease ablation (must recover arm2 exactly)
// Value comes from functional shadow memory, so BITIDENTICAL is structural; the
// arm changes only WHICH reads re-fetch (traffic counters) and latency.
//
// Args: <num_clusters> <nepochs> <tile_lines> <f_permille> [change_period_S]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8
#endif
#ifndef MAX_EPOCHS
#define MAX_EPOCHS 512
#endif
#define LINE_BYTES 128
#define FLOATS_PER_LINE (LINE_BYTES/4)

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// A volatile side effect is REQUIRED so nvcc/ptxas does not optimize the call
// away (an empty body gets eliminated -> the sim never sees the registration).
__device__ volatile int g_reg_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_reg_flag = (int)((unsigned long long)addr & 1); (void)addr;
}

__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)LINE_BYTES - 1);
}
// warp0 registers every 128B line of [base, base+bytes) as a marked comm line.
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

// Deterministic value of table line `ln` float `i` at its current "generation"
// gen (number of times it has been rewritten). Shared by device + host ref.
__host__ __device__ __forceinline__ int cell_val(int cid,int ln,int gen,int i){
  return (int)(((unsigned)(cid*2654435761u) ^ (unsigned)(ln*2246822519u)
              ^ (unsigned)(gen*40503u) ^ (unsigned)(i*668265263u)) & 0x7fffffff);
}
// Does table line `ln` get rewritten at epoch e (e>=1)?  A line is "dynamic"
// (participates in change) iff (hash(ln) % 1000) < f_permille; a dynamic line
// changes only every S epochs (epoch-survival). Deterministic, host-matchable.
__host__ __device__ __forceinline__ int changes_at(int ln,int e,int f_permille,int S){
  if (e < 1) return 0;
  unsigned h = ((unsigned)ln*2654435761u) % 1000u;
  if ((int)h >= f_permille) return 0;      // stationary line
  return ((e % S) == 0) ? 1 : 0;           // dynamic line, period S
}
// Generation of line `ln` at the START of epoch e (how many rewrites so far).
__host__ __device__ __forceinline__ int gen_at(int ln,int e,int f_permille,int S){
  int g=0; for(int k=1;k<=e;++k) g+=changes_at(ln,k,f_permille,S); return g;
}

// table arena: g_tbl[ (cid*tile_lines + ln)*FLOATS_PER_LINE + i ]
template<int CS>
__global__ void __cluster_dims__(CS,1,1)
lease_kernel(int* __restrict__ g_tbl, int num_clusters, int nepochs,
             int tile_lines, int f_permille, int S){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  const int tf   = tile_lines * FLOATS_PER_LINE;
  size_t tbl_base = (size_t)cid * tile_lines * FLOATS_PER_LINE;

  // Every rank registers the whole table as marked comm lines (all ranks of a
  // CGA share the same (gpc,cluster_slot) key, so duplicates are harmless).
  if (tid < 32) register_range_warp0(g_tbl + tbl_base, (size_t)tf*sizeof(int));
  __syncthreads();
  cl.sync();

  long long acc = 0;
  for (int e = 0; e < nepochs; ++e) {
    // PRODUCER (rank 0): epoch 0 writes ALL lines; later epochs rewrite only
    // the lines whose change schedule fires this epoch.
    if (rank == 0) {
      for (int ln = 0; ln < tile_lines; ++ln) {
        if (e == 0 || changes_at(ln, e, f_permille, S)) {
          int gen = gen_at(ln, e, f_permille, S);
          int* line = g_tbl + tbl_base + (size_t)ln*FLOATS_PER_LINE;
          for (int i = tid; i < FLOATS_PER_LINE; i += blockDim.x)
            st_wb_s32(&line[i], cell_val(cid, ln, gen, i));
        }
      }
    }
    // Two cluster.syncs per epoch: the FIRST orders producer-writes-before
    // consumer-reads (functional correctness -- avoids the WAR race), the
    // SECOND closes the epoch. So one benchmark-epoch == TWO lease ticks; the
    // change schedule and the lease budget are both expressed in EPOCHS and the
    // sim-side budget is passed pre-doubled by the harness (RESC_LEASE_S=2*S).
    cl.sync();  // producer writes visible before consumer reads
    // CONSUMERS (rank 1..CS-1): re-read the ENTIRE table every epoch.
    if (rank > 0) {
      for (int ln = 0; ln < tile_lines; ++ln) {
        int* line = g_tbl + tbl_base + (size_t)ln*FLOATS_PER_LINE;
        for (int i = tid; i < FLOATS_PER_LINE; i += blockDim.x)
          acc += ld_ca_s32(&line[i]);
      }
    }
    cl.sync();  // epoch boundary (lease may expire / arm2 self-invalidates)
  }
  if (acc) atomicAdd(&g_checksum, (unsigned long long)acc);
#endif
}

int main(int argc, char** argv){
  int num_clusters = (argc>1)? atoi(argv[1]) : 4;
  int nepochs      = (argc>2)? atoi(argv[2]) : 8;
  int tile_lines   = (argc>3)? atoi(argv[3]) : 4;
  int f_permille   = (argc>4)? atoi(argv[4]) : 0;    // 0..1000
  int S            = (argc>5)? atoi(argv[5]) : 1;    // change period (epochs)
  if (num_clusters<1) num_clusters=1;
  if (nepochs<1) nepochs=1; if (nepochs>MAX_EPOCHS) nepochs=MAX_EPOCHS;
  if (tile_lines<1) tile_lines=1;
  if (f_permille<0) f_permille=0; if (f_permille>1000) f_permille=1000;
  if (S<1) S=1;
  const int CS=CLUSTER_SIZE;
  const int tf=tile_lines*FLOATS_PER_LINE;

  size_t ntot=(size_t)num_clusters*tile_lines*FLOATS_PER_LINE;
  int* d=nullptr;
  CUDA_CHECK(cudaMalloc(&d, ntot*sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, ntot*sizeof(int)));
  dim3 grid(CS*num_clusters,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  lease_kernel<CLUSTER_SIZE><<<grid,block>>>(d, num_clusters, nepochs,
                                             tile_lines, f_permille, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long dev=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&dev, g_checksum, sizeof(dev)));
  // reference: each consumer rank (1..CS-1) sums ALL lines each epoch; the line
  // value at epoch e uses generation gen_at(ln,e,...) (== value produced at the
  // most recent rewrite <= e).
  unsigned long long ref=0;
  for (int cid=0; cid<num_clusters; ++cid)
    for (int r=1;r<CS;++r)
      for (int e=0;e<nepochs;++e)
        for (int ln=0; ln<tile_lines; ++ln){
          int gen=gen_at(ln,e,f_permille,S);
          long long s=0;
          for (int i=0;i<FLOATS_PER_LINE;++i) s+=cell_val(cid,ln,gen,i);
          ref+=(unsigned long long)s;
        }
  printf("LEASE_BENCH cluster=%d num_clusters=%d nepochs=%d tile_lines=%d "
         "f_permille=%d S=%d OUT_CHECKSUM=%llu REF_CHECKSUM=%llu BITIDENTICAL=%d\n",
         CS, num_clusters, nepochs, tile_lines, f_permille, S, dev, ref,
         (dev==ref)?1:0);
  CUDA_CHECK(cudaFree(d));
  return 0;
}
