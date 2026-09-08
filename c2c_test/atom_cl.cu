#include <cstdio>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;
__global__ void __cluster_dims__(8,1,1) k(int* a, int rounds){
  cg::cluster_group cl = cg::this_cluster();
  int cid = blockIdx.x/8;
  int* my = a + (size_t)cid*32;  // 128B per cluster (32 int)
  for(int r=0;r<rounds;++r){
    if(threadIdx.x==0) atomicAdd(&my[0], 1);
    cl.sync();
  }
}
int main(){ int nc=2,rounds=5; int* d; cudaMalloc(&d,(size_t)nc*32*4); cudaMemset(d,0,(size_t)nc*32*4);
  k<<<8*nc,64>>>(d,rounds); cudaDeviceSynchronize();
  int h[2]; cudaMemcpy(h,d,4,cudaMemcpyDeviceToHost); cudaMemcpy(h+1,d+32,4,cudaMemcpyDeviceToHost);
  printf("ATOM_CL c0=%d c1=%d (expect %d each)\n", h[0],h[1], 8*rounds); return 0; }
