// fe_probe.cu -- READ-ONLY frontend feasibility probe (no simulator source touched).
// Asks: which of the constructs needed to model BOTH cluster-barrier forms can
// ClusterSim's PTX frontend actually parse?
//   VARIANT 0 : plain cluster.sync()                              (control, known good)
//   VARIANT 1 : fence.acquire.sync_restrict::shared::cluster.cluster  (the ISA escape)
//   VARIANT 2 : relaxed cluster-scope atomic on .shared::cluster  (software-barrier primitive)
//   VARIANT 3 : barrier.cluster.arrive/wait written as raw inline asm
#include <cstdio>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef VARIANT
#define VARIANT 0
#endif
#ifndef CS
#define CS 2
#endif

__device__ __forceinline__ float ld_ca_f32(const float* p){ float v;
  asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p)); return v; }

__global__ __cluster_dims__(CS,1,1)
void probe(const float* __restrict__ tile, float* __restrict__ out, int n)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  __shared__ unsigned smem_flag[32];
  if (threadIdx.x < 32) smem_flag[threadIdx.x] = 0u;
  __syncthreads();
  float acc = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += ld_ca_f32(&tile[i]);

#if VARIANT == 0
  cl.sync();
#elif VARIANT == 1
  asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
#elif VARIANT == 2
  {
    unsigned* q = smem_flag;
    unsigned old = 0u;
    asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0,[%1],1;"
                 : "=r"(old) : "l"(q) : "memory");
    acc += (float)(old & 1u);
  }
#elif VARIANT == 3
  asm volatile("barrier.cluster.arrive;" ::: "memory");
  asm volatile("barrier.cluster.wait;"   ::: "memory");
#endif

  for (int i = threadIdx.x; i < n; i += blockDim.x) acc += ld_ca_f32(&tile[i]);
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
#endif
}

int main(){
  const int n = 1024, nb = CS*2;
  float *dt=nullptr,*doo=nullptr;
  cudaMalloc(&dt, n*sizeof(float)); cudaMalloc(&doo, nb*sizeof(float));
  cudaMemset(dt,0,n*sizeof(float));
  probe<<<nb,128>>>(dt,doo,n);
  cudaError_t e = cudaDeviceSynchronize();
  printf("FE_PROBE VARIANT=%d launch=%s\n", VARIANT, cudaGetErrorString(e));
  return 0;
}
