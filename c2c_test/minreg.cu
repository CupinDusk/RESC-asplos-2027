// minreg.cu -- minimal RESC registration reproducer. Isolates whether the sim's
// gpc-slot lookup in gpgpusim_register_cluster_coherent_addr crashes for a plain
// cluster kernel. Mirrors resc_e2e's registration exactly.
#include <cstdio>
#include <cstdint>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;
#define RESC_LINE_BYTES 128
#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
__device__ volatile int g_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){ g_flag=(int)((unsigned long long)addr&1); (void)addr; }
__device__ __forceinline__ uintptr_t adl(uintptr_t x){return x&~((uintptr_t)RESC_LINE_BYTES-1);}
__device__ __forceinline__ void reg_range(void* base,size_t bytes){
  uintptr_t b=(uintptr_t)base,s=adl(b);
  int nl=(int)(((b+bytes-s)+RESC_LINE_BYTES-1)/RESC_LINE_BYTES);
  int lane=threadIdx.x&31;
  for(int off=0;off<nl;off+=32){int idx=off+lane; uintptr_t line=s+(uintptr_t)((idx<nl)?idx:0)*RESC_LINE_BYTES; gpgpusim_register_cluster_coherent_addr((void*)line);}
}
template<int CS> __global__ __cluster_dims__(CS,1,1)
void k(float* g, float* out){
#if __CUDA_ARCH__>=900
  cg::cluster_group cl=cg::this_cluster(); int rank=cl.block_rank(); int tid=threadIdx.x;
  int cid=(int)(blockIdx.x/CS);
  float* slab=g+(size_t)cid*4096;
  if(rank==0 && CS>0){ if(tid<32){ reg_range(slab, 512); } __syncwarp(); }
  cl.sync();
  if(tid==0) out[(size_t)cid*CS+rank]=slab[0]+1.0f;
#endif
}
int main(int argc,char**argv){
  int cs=argc>1?atoi(argv[1]):8; int ncl=argc>2?atoi(argv[2]):1;
  float* pad; CUDA_CHECK(cudaMalloc(&pad,65536));
  float* g; CUDA_CHECK(cudaMalloc(&g,(size_t)ncl*4096*sizeof(float)));
  float* out; CUDA_CHECK(cudaMalloc(&out,(size_t)ncl*cs*sizeof(float)));
  CUDA_CHECK(cudaMemset(g,0,(size_t)ncl*4096*sizeof(float)));
  dim3 grid(cs*ncl),block(256);
  switch(cs){case 2:k<2><<<grid,block>>>(g,out);break;case 4:k<4><<<grid,block>>>(g,out);break;case 8:k<8><<<grid,block>>>(g,out);break;}
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  printf("MINREG_OK cs=%d ncl=%d\n",cs,ncl);
  return 0;
}
