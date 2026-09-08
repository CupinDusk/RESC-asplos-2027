// RESC OVERLAPPING-EPOCH chain-pipeline benchmark (distributed producers).
//
// Unlike pipe_overlap (one producer -> CS-1 consumers, which funnels every
// read-share through ONE injection port = the injection cliff), here EACH rank
// r produces its own tile every epoch and consumes its LEFT neighbor (r-1)'s
// tile produced LAG epochs earlier. Fan-in per producer = 1, so the read-share
// traffic is spread across CS-1 distinct supplier injection ports -- the regime
// where overlapping leases can actually pay off under the realistic model.
//
// A consumer read that lags by LAG epochs is served over the fabric IFF the
// producing generation is still within the ring window => needs G >= LAG+1.
//   G=1 (eager/hLRC): every lagged read drains to L2 (pipeline serialized).
//   G>=LAG+1        : lagged reads served peer-L1 (stages overlap).
// Byte-identical binary across arms; discipline chosen only by RESC_GEN_RING.
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
#define FLOATS_PER_LINE (LINE_BYTES/4)

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

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

__host__ __device__ __forceinline__ int tile_val(int cid,int rank,int gen,int i){
  return (int)(((unsigned)(cid*2654435761u) ^ (unsigned)(rank*2246822519u)
              ^ (unsigned)(gen*40503u) ^ (unsigned)(i*668265263u)) & 0x7fffffff);
}

// arena: g_tiles[ ((cid*CS + rank)*nstages + gen)*tile_floats + i ]
template<int CS>
__global__ void __cluster_dims__(CS,1,1)
chain_kernel(int* __restrict__ g_tiles, int num_clusters, int nstages,
             int lag, int tile_lines){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  const int tf   = tile_lines * FLOATS_PER_LINE;
  size_t rank_base = (((size_t)cid*CS + rank)*nstages) * (size_t)tf;

  // each rank registers ITS OWN tile lines (distributed producers).
  if (tid < 32) register_range_warp0(g_tiles + rank_base, (size_t)nstages*tf*sizeof(int));
  __syncthreads();
  cl.sync();

  long long acc = 0;
  for (int e = 0; e < nstages; ++e) {
    // produce my own generation-e tile
    int* mine = g_tiles + rank_base + (size_t)e*tf;
    for (int i = tid; i < tf; i += blockDim.x)
      st_wb_s32(&mine[i], tile_val(cid, rank, e, i));
    // consume LEFT neighbor (rank-1)'s tile from epoch (e-lag)
    if (rank > 0 && e >= lag) {
      int g = e - lag;
      size_t nbr_base = (((size_t)cid*CS + (rank-1))*nstages) * (size_t)tf;
      int* nbr = g_tiles + nbr_base + (size_t)g*tf;
      for (int i = tid; i < tf; i += blockDim.x)
        acc += ld_ca_s32(&nbr[i]);
    }
    cl.sync();  // epoch tick
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
  if (nstages<1) nstages=1; if (nstages>MAX_STAGES) nstages=MAX_STAGES;
  if (lag<0) lag=0; if (tile_lines<1) tile_lines=1;
  const int CS=CLUSTER_SIZE;
  const int tf=tile_lines*FLOATS_PER_LINE;

  size_t ntot=(size_t)num_clusters*CS*nstages*tf;
  int* d=nullptr;
  CUDA_CHECK(cudaMalloc(&d, ntot*sizeof(int)));
  CUDA_CHECK(cudaMemset(d, 0, ntot*sizeof(int)));
  dim3 grid(CS*num_clusters,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  chain_kernel<CLUSTER_SIZE><<<grid,block>>>(d, num_clusters, nstages, lag, tile_lines);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long dev=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&dev, g_checksum, sizeof(dev)));
  // reference: rank r (1..CS-1) reads rank(r-1) gen(e-lag), e=lag..nstages-1
  unsigned long long ref=0;
  for (int cid=0; cid<num_clusters; ++cid)
    for (int r=1;r<CS;++r)
      for (int e=lag;e<nstages;++e){
        int g=e-lag; long long s=0;
        for (int i=0;i<tf;++i) s+=tile_val(cid,r-1,g,i);
        ref+=(unsigned long long)s;
      }
  printf("PIPE_CHAIN cluster=%d num_clusters=%d nstages=%d lag=%d tile_lines=%d "
         "OUT_CHECKSUM=%llu REF_CHECKSUM=%llu BITIDENTICAL=%d\n",
         CS, num_clusters, nstages, lag, tile_lines, dev, ref, (dev==ref)?1:0);
  CUDA_CHECK(cudaFree(d));
  return 0;
}
