// RESC OVERLAPPING-EPOCH pipeline benchmark (ASPLOS center-of-gravity novelty).
//
// A cluster runs a software pipeline: rank 0 is a PRODUCER that streams one
// marked communication tile per epoch; ranks 1..CS-1 are CONSUMERS that each
// read a tile produced LAG epochs earlier (a stage behind). Each epoch is
// closed by cluster.sync() -- the RESC epoch tick.
//
// Because tiles are DISTINCT per generation, every consumer read is a first
// touch => misses the consumer L1 => takes the RESC peer-L1 read-share path,
// served from the producer's L1 over the SM-to-SM fabric IFF the tile's
// producing generation is still inside the live generation-ring window.
//
//   To serve a read that lags by LAG epochs you need ring depth G >= LAG+1:
//     LAG=1  served by G>=2  (DeNovo current+next can do this)
//     LAG=2  served by G>=3  (DeNovo G=2 CANNOT -> the new regime)
//     LAG=3  served by G>=4
//   G=1 (eager/hLRC bulk self-invalidate at each sync) serves NOTHING lagged:
//   every lagged read drains to L2. That is the pipeline serialization the
//   overlapping ring removes.
//
// The coherence discipline is chosen ENTIRELY by the sim env RESC_GEN_RING (=G);
// the compiled binary is byte-identical across arms. Functional value always
// comes from L2 by the write-through invariant, so OUT_CHECKSUM must be equal
// across all G (bit-identity of OVERLAP vs EAGER).
//
// Args: <num_clusters> <nstages> [lag] [tile_lines]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8
#endif
#ifndef MAX_STAGES
#define MAX_STAGES 512
#endif
#define LINE_BYTES 128
#define FLOATS_PER_LINE (LINE_BYTES/4)   // 32

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// simulator hook (no-op body so real HW links+runs; intercepted functionally)
__device__ volatile int g_reg_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_reg_flag = (int)((unsigned long long)addr & 1); (void)addr;
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)LINE_BYTES - 1);
}
// warp0 registers every 128B line spanning [base, base+bytes) (proven idiom).
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

__host__ __device__ __forceinline__ int tile_val(int cid,int gen,int i){
  return (int)(((unsigned)(cid*2654435761u) ^ (unsigned)(gen*40503u) ^ (unsigned)(i*2246822519u)) & 0x7fffffff);
}

template<int CS>
__global__ void __cluster_dims__(CS,1,1)
pipe_kernel(int* __restrict__ g_tiles, int num_clusters, int nstages,
            int lag, int tile_lines){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  const int tile_floats = tile_lines * FLOATS_PER_LINE;
  // Per-cluster arena: one distinct tile per stage (distinct address per gen ->
  // every consumer read is a first-touch miss -> exercises the read-share path).
  // layout: g_tiles[ (cid*nstages + gen) * tile_floats + i ]
  size_t cluster_base = (size_t)cid * (size_t)nstages * (size_t)tile_floats;

  // rank 0 registers all of this cluster's tile lines as RESC comm lines.
  if (rank == 0 && tid < 32) {
    register_range_warp0(g_tiles + cluster_base,
                         (size_t)nstages * tile_floats * sizeof(int));
  }
  __syncthreads();
  cl.sync();   // epoch boundary (registration complete)

  long long acc = 0;
  for (int e = 0; e < nstages; ++e) {
    // ---- PRODUCE generation e ----
    if (rank == 0) {
      int* tile = g_tiles + cluster_base + (size_t)e * tile_floats;
      for (int i = tid; i < tile_floats; i += blockDim.x)
        st_wb_s32(&tile[i], tile_val(cid, e, i));
    }
    // ---- CONSUME generation (e-lag), a stage behind ----
    if (rank != 0 && e >= lag) {
      int g = e - lag;
      int* tile = g_tiles + cluster_base + (size_t)g * tile_floats;
      // each consumer reads the full lagged tile (fan-out over the fabric)
      for (int i = tid; i < tile_floats; i += blockDim.x)
        acc += ld_ca_s32(&tile[i]);
    }
    cl.sync();   // epoch tick: advance the generation ring
  }
  if (acc) atomicAdd(&g_checksum, (unsigned long long)acc);
#endif
}

int main(int argc, char** argv){
  int num_clusters = (argc>1)? atoi(argv[1]) : 8;
  int nstages      = (argc>2)? atoi(argv[2]) : 32;
  int lag          = (argc>3)? atoi(argv[3]) : 2;
  int tile_lines   = (argc>4)? atoi(argv[4]) : 4;
  if (num_clusters<1) num_clusters=1;
  if (nstages<1) nstages=1;
  if (nstages>MAX_STAGES) nstages=MAX_STAGES;
  if (lag<0) lag=0;
  if (tile_lines<1) tile_lines=1;
  const int CS=CLUSTER_SIZE;
  const int tile_floats = tile_lines*FLOATS_PER_LINE;

  size_t ntot = (size_t)num_clusters*nstages*tile_floats;
  int* d_tiles=nullptr;
  CUDA_CHECK(cudaMalloc(&d_tiles, ntot*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tiles, 0, ntot*sizeof(int)));

  dim3 grid(CS*num_clusters,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  pipe_kernel<CLUSTER_SIZE><<<grid,block>>>(d_tiles, num_clusters, nstages, lag, tile_lines);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long dev=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&dev, g_checksum, sizeof(dev)));

  // bit-exact CPU reference: each consumer (CS-1 of them) reads tiles gen 0..nstages-1-lag
  unsigned long long ref=0;
  for (int cid=0; cid<num_clusters; ++cid)
    for (int e=lag; e<nstages; ++e){
      int g=e-lag; long long s=0;
      for (int i=0;i<tile_floats;++i) s += tile_val(cid,g,i);
      ref += (unsigned long long)((CS-1)*s);
    }

  printf("PIPE_OVERLAP cluster=%d num_clusters=%d nstages=%d lag=%d tile_lines=%d "
         "OUT_CHECKSUM=%llu REF_CHECKSUM=%llu BITIDENTICAL=%d\n",
         CS, num_clusters, nstages, lag, tile_lines, dev, ref, (dev==ref)?1:0);
  CUDA_CHECK(cudaFree(d_tiles));
  return 0;
}
