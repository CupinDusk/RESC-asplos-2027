#include <cstdio>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){ asm volatile("" :: "l"(addr) : "memory"); }
__global__ void __cluster_dims__(8,1,1) k(long long* a, int rounds){
  cg::cluster_group cl = cg::this_cluster();
  int rank = cl.block_rank();
  int cid = blockIdx.x/8;
  long long* my = a + (size_t)cid*16;   // 128B per cluster
  if(rank==0 && threadIdx.x<32){
    int lane=threadIdx.x&31;
    if(lane==0) gpgpusim_register_cluster_coherent_addr((void*)my);
  }
  cl.sync();
  int* a32 = (int*)my;
  for(int r=0;r<rounds;++r){
    if(threadIdx.x==0) atomicAdd(&a32[0], ((cid*131+rank*17+r*7+1)&0xFFFF));
    cl.sync();
  }
}
int main(){ int nc=2,rounds=3; long long* d; cudaMalloc(&d,(size_t)nc*16*8); cudaMemset(d,0,(size_t)nc*16*8);
  k<<<8*nc,64>>>(d,rounds); cudaDeviceSynchronize();
  int h[2]; cudaMemcpy(h,d,4,cudaMemcpyDeviceToHost); cudaMemcpy(h+1,(char*)d+128,4,cudaMemcpyDeviceToHost);
  printf("ATOM_CL2 c0=%d c1=%d\n", h[0],h[1]); return 0; }
