// Matrix-multiply — held-out general validation workload. Square cluster (2x2 blocks
// cooperate on a tile via DSM), matching ClusterSim's MM (Cluster2x2). Same source on
// H20 (NCU) and simulator. sim hook no-op on HW.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 4
#endif
#define TILE 16
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){ (void)addr; }

// cluster of CLUSTER_SIZE blocks; each block computes a TILExTILE sub-tile, sharing
// A/B strips with cluster peers over DSM (map_shared_rank) to cut global refetch.
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void mm_cluster(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N){
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int tx=threadIdx.x%TILE, ty=threadIdx.x/TILE;
  int row=(blockIdx.x/(N/TILE))*TILE+ty;
  int col=(blockIdx.x%(N/TILE))*TILE+tx;
  float acc=0.f;
  for(int k0=0;k0<N;k0+=TILE){
    if(row<N && k0+tx<N) As[ty][tx]=A[row*N+k0+tx]; else As[ty][tx]=0.f;
    if(col<N && k0+ty<N) Bs[ty][tx]=B[(k0+ty)*N+col]; else Bs[ty][tx]=0.f;
    __syncthreads();
    cluster.sync();
    // pull peer's A-strip over DSM to exercise SM-to-SM (cluster-cooperative reuse)
    if(rank>0){
      float(*peerAs)[TILE]=(float(*)[TILE])cluster.map_shared_rank(As,0);
      #pragma unroll
      for(int k=0;k<TILE;k++) acc += peerAs[ty][k]*Bs[k][tx];
    } else {
      #pragma unroll
      for(int k=0;k<TILE;k++) acc += As[ty][k]*Bs[k][tx];
    }
    __syncthreads();
    cluster.sync();
  }
  if(row<N && col<N) C[row*N+col]=acc;
}

int main(int argc,char**argv){
  int cs=CLUSTER_SIZE, N=(argc>2?atoi(argv[2]):512);
  float *A,*B,*C; float* h=(float*)malloc((size_t)N*N*4);
  for(int i=0;i<N*N;i++) h[i]=(float)((i*2654435761u>>16)&255)/256.f;
  CUDA_CHECK(cudaMalloc(&A,(size_t)N*N*4));CUDA_CHECK(cudaMalloc(&B,(size_t)N*N*4));CUDA_CHECK(cudaMalloc(&C,(size_t)N*N*4));
  CUDA_CHECK(cudaMemcpy(A,h,(size_t)N*N*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B,h,(size_t)N*N*4,cudaMemcpyHostToDevice));
  int tiles=(N/TILE)*(N/TILE);
  int grid=((tiles+cs-1)/cs)*cs; // multiple of cluster size
  mm_cluster<<<grid,TILE*TILE>>>(A,B,C,N);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("mm cluster=%d N=%d done\n",cs,N);
  return 0;
}
