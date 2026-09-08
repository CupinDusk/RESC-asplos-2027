// RESC OVERLAPPING-EPOCH fan-out crossover benchmark.
//
// Generalizes the pipeline to a TUNABLE consumers-per-supplier ratio so we can
// map the injection-concentration crossover under the realistic (charge-ON)
// model. A cluster of CS blocks is split into P producer groups; the first P
// ranks are producers, the rest are consumers. Consumer c is assigned to
// producer (c % P), so each producer serves ~ (CS-P)/P consumers = "fan".
// Each producer streams one distinct marked tile per epoch; each consumer reads
// ITS producer's tile from LAG epochs earlier. Epoch closed by cluster.sync.
//
//   fan (consumers-per-supplier) = ceil((CS - P) / P)
//   P=1  -> fan = CS-1  (all consumers on one egress port: worst case)
//   P=CS/2 (=4 at CS=8) -> fan = 1  (distributed)
//
// Optional compute tile: each rank does COMPUTE_ITERS fused-multiply-adds per
// epoch on a register accumulator (models real per-stage compute so freed
// overlap/occupancy can show). O(1) SMEM touch keeps a dynamic tile live for
// occupancy if TILE_KB>0.
//
// Byte-identical across G (coherence chosen only by RESC_GEN_RING); OUT_CHECKSUM
// must match across all G.
//
// Args: <num_clusters> <nstages> <lag> <tile_lines> <nproducers> <compute_iters>
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
#ifndef TILE_KB
#define TILE_KB 0
#endif
#define LINE_BYTES 128
#define FLOATS_PER_LINE (LINE_BYTES/4)

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

__device__ volatile int g_reg_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_reg_flag=(int)((unsigned long long)addr&1); (void)addr;
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)LINE_BYTES-1);
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

__host__ __device__ __forceinline__ int tile_val(int cid,int prod,int gen,int i){
  return (int)(((unsigned)(cid*2654435761u) ^ (unsigned)(prod*2246822519u)
              ^ (unsigned)(gen*40503u) ^ (unsigned)(i*668265263u)) & 0x7fffffff);
}

// arena: g_tiles[ ((cid*P + prod)*nstages + gen)*tf + i ]
template<int CS>
__global__ void __cluster_dims__(CS,1,1)
fanout_kernel(int* __restrict__ g_tiles, int num_clusters, int nstages,
              int lag, int tile_lines, int nprod, int compute_iters){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  const int tf   = tile_lines*FLOATS_PER_LINE;
  const int P    = nprod;

  extern __shared__ float s_tile[];
  const int TILE_ELEMS=(TILE_KB*1024)/(int)sizeof(float);
  if (TILE_ELEMS>0 && tid==0){ s_tile[0]=(float)rank; s_tile[TILE_ELEMS-1]=(float)rank; }

  const bool is_prod = (rank < P);
  const int  myprod  = is_prod ? rank : (rank % P);   // consumer's assigned producer
  size_t prod_base   = (((size_t)cid*P + myprod)*nstages) * (size_t)tf;

  // producers register their own tile lines
  if (is_prod && tid<32)
    register_range_warp0(g_tiles + prod_base, (size_t)nstages*tf*sizeof(int));
  __syncthreads();
  cl.sync();

  long long acc=0; float cacc=1.0f;
  for (int e=0;e<nstages;++e){
    if (is_prod){
      int* mine = g_tiles + prod_base + (size_t)e*tf;
      for (int i=tid;i<tf;i+=blockDim.x) st_wb_s32(&mine[i], tile_val(cid,myprod,e,i));
    } else if (e>=lag){
      int g=e-lag;
      int* t = g_tiles + prod_base + (size_t)g*tf;
      for (int i=tid;i<tf;i+=blockDim.x) acc += ld_ca_s32(&t[i]);
    }
    // per-stage compute (models real work so freed overlap can show)
    #pragma unroll 1
    for (int k=0;k<compute_iters;++k) cacc = cacc*1.0000001f + 0.5f;
    cl.sync();  // epoch tick
  }
  if (cacc==-98765.f) g_tiles[0]=(int)cacc;   // prevent DCE
  if (acc) atomicAdd(&g_checksum, (unsigned long long)acc);
#endif
}

int main(int argc, char** argv){
  int num_clusters=(argc>1)?atoi(argv[1]):4;
  int nstages     =(argc>2)?atoi(argv[2]):32;
  int lag         =(argc>3)?atoi(argv[3]):2;
  int tile_lines  =(argc>4)?atoi(argv[4]):4;
  int nprod       =(argc>5)?atoi(argv[5]):1;
  int compute_iters=(argc>6)?atoi(argv[6]):0;
  if(num_clusters<1)num_clusters=1;
  if(nstages<1)nstages=1; if(nstages>MAX_STAGES)nstages=MAX_STAGES;
  if(lag<0)lag=0; if(tile_lines<1)tile_lines=1;
  const int CS=CLUSTER_SIZE;
  if(nprod<1)nprod=1; if(nprod>CS-1)nprod=CS-1;   // need >=1 consumer
  const int tf=tile_lines*FLOATS_PER_LINE;

  size_t ntot=(size_t)num_clusters*nprod*nstages*tf;
  int* d=nullptr;
  CUDA_CHECK(cudaMalloc(&d, ntot*sizeof(int)));
  CUDA_CHECK(cudaMemset(d,0,ntot*sizeof(int)));
  int smem = TILE_KB*1024;
  dim3 grid(CS*num_clusters,1,1), block(64,1,1);
  CUDA_CHECK(cudaDeviceSynchronize());
  fanout_kernel<CLUSTER_SIZE><<<grid,block,smem>>>(d,num_clusters,nstages,lag,tile_lines,nprod,compute_iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long dev=0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&dev, g_checksum, sizeof(dev)));
  // reference: each consumer rank (P..CS-1) reads producer (rank%P) gen(e-lag)
  unsigned long long ref=0;
  for(int cid=0;cid<num_clusters;++cid)
    for(int r=nprod;r<CS;++r){
      int prod=r%nprod;
      for(int e=lag;e<nstages;++e){ int g=e-lag; long long s=0;
        for(int i=0;i<tf;++i) s+=tile_val(cid,prod,g,i);
        ref+=(unsigned long long)s; }
    }
  int fan = (CS-nprod + nprod-1)/nprod;  // ceil consumers-per-producer (approx)
  printf("PIPE_FANOUT cluster=%d num_clusters=%d nstages=%d lag=%d tile_lines=%d "
         "nprod=%d fan=%d compute_iters=%d tile_kb=%d "
         "OUT_CHECKSUM=%llu REF_CHECKSUM=%llu BITIDENTICAL=%d\n",
         CS,num_clusters,nstages,lag,tile_lines,nprod,fan,compute_iters,TILE_KB,
         dev,ref,(dev==ref)?1:0);
  CUDA_CHECK(cudaFree(d));
  return 0;
}
