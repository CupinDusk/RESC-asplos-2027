// Histogram — held-out general validation workload for H20 sim validation.
// Cluster-cooperative binning: blocks in a cluster accumulate into DSM-shared bins,
// exercising the DSMEM/SM-to-SM path (irregular scatter). Same source runs on real
// H20 (NCU cycle) and on the ClusterSim simulator. sim hook is a no-op on HW.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 2
#endif
#define NBINS 256

// simulator hook (no-op body so real HW links + runs)
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){ (void)addr; }

__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void hist_cluster(const unsigned* __restrict__ in, int N, unsigned* __restrict__ out){
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  int tid = threadIdx.x;
  __shared__ unsigned local[NBINS];
  for(int b=tid;b<NBINS;b+=blockDim.x) local[b]=0;
  __syncthreads();
  // each block strides over its slice, accumulating into its own shared bins
  int gid = (blockIdx.x*blockDim.x)+tid;
  int stride = gridDim.x*blockDim.x;
  for(int i=gid;i<N;i+=stride){ atomicAdd(&local[in[i]&(NBINS-1)],1u); }
  __syncthreads();
  cluster.sync();
  // block 0 of cluster reduces peers' shared bins over SM-to-SM (DSM), then writes global
  if(rank==0){
    for(int r=1;r<CLUSTER_SIZE;r++){
      unsigned* peer = (unsigned*)cluster.map_shared_rank(local, r);
      for(int b=tid;b<NBINS;b+=blockDim.x) atomicAdd(&local[b], peer[b]);
    }
    __syncthreads();
    for(int b=tid;b<NBINS;b+=blockDim.x) atomicAdd(&out[b], local[b]);
  }
  cluster.sync();
}

int main(int argc,char**argv){
  int cluster_size=CLUSTER_SIZE, N=(argc>2?atoi(argv[2]):(1<<22));
  unsigned *in,*out; unsigned* h=(unsigned*)malloc(N*4);
  for(int i=0;i<N;i++) h[i]=(unsigned)((i*2654435761u)>>8);
  CUDA_CHECK(cudaMalloc(&in,(size_t)N*4)); CUDA_CHECK(cudaMalloc(&out,NBINS*4));
  CUDA_CHECK(cudaMemcpy(in,h,(size_t)N*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(out,0,NBINS*4));
  dim3 grid(cluster_size* ( (argc>3)?atoi(argv[3]):66 )); dim3 block(256);
  hist_cluster<<<grid,block>>>(in,N,out);
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("hist cluster=%d N=%d done\n",cluster_size,N);
  return 0;
}
