// Real-H20 DSM-vs-global crossover microbench: one producer (rank0) publishes a
// per-cluster payload of `nfloats`, consumers read it via DSM (map_shared_rank) vs
// via global. Times the CONSUMER read with clock64, N repeats for CI.
// Pure CUDA, no torch/DGL. arch sm_90.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 8
#endif
#ifndef NFLOATS
#define NFLOATS 256
#endif
#define MAXF 2048

__device__ unsigned long long g_dsm_cyc[1024];
__device__ unsigned long long g_glb_cyc[1024];

// DSM arm: rank0 fills its SMEM payload; consumers read rank0's SMEM via map_shared_rank.
__global__ void __cluster_dims__(CLUSTER_SIZE,1,1)
dsm_kernel(int nfloats, int iters) {
  __shared__ float s_payload[MAXF];
  cg::cluster_group cl = cg::this_cluster();
  unsigned rank = cl.block_rank();
  int tid = threadIdx.x;
  // rank0 produces
  if (rank == 0) for (int i=tid;i<nfloats;i+=blockDim.x) s_payload[i]=(float)(i*3+1);
  cl.sync();
  float* src = (rank==0)? s_payload : (float*)cl.map_shared_rank(s_payload, 0);
  // consumers (rank>0) time the read
  unsigned long long t0=0,t1=0; float acc=0.f;
  if (rank!=0 && tid==0) t0=clock64();
  if (rank!=0) {
    for (int it=0; it<iters; ++it)
      for (int i=tid;i<nfloats;i+=blockDim.x) acc+=src[i];
  }
  if (rank!=0 && tid==0) { t1=clock64();
    unsigned slot=(blockIdx.x)&1023; g_dsm_cyc[slot]=(t1>t0)?(t1-t0):0ULL;
  }
  cl.sync(); // keep all ranks resident until reads complete (avoids use-after-free)
  if (acc==-12345.f) s_payload[0]=acc;
}

// Global arm: rank0 writes payload to global; consumers read from global.
__global__ void __cluster_dims__(CLUSTER_SIZE,1,1)
global_kernel(float* g_payload, int nfloats, int iters) {
  cg::cluster_group cl = cg::this_cluster();
  unsigned rank = cl.block_rank();
  int tid = threadIdx.x;
  int cluster_id = blockIdx.x / CLUSTER_SIZE;
  float* base = g_payload + (size_t)cluster_id * MAXF;
  if (rank==0) for (int i=tid;i<nfloats;i+=blockDim.x) base[i]=(float)(i*3+1);
  cl.sync();
  __threadfence();
  unsigned long long t0=0,t1=0; float acc=0.f;
  if (rank!=0 && tid==0) t0=clock64();
  if (rank!=0) {
    for (int it=0; it<iters; ++it)
      for (int i=tid;i<nfloats;i+=blockDim.x) acc+=base[i];
  }
  if (rank!=0 && tid==0) { t1=clock64();
    unsigned slot=(blockIdx.x)&1023; g_glb_cyc[slot]=(t1>t0)?(t1-t0):0ULL;
  }
  cl.sync();
  if (acc==-12345.f) base[0]=acc;
}

int main(int argc, char** argv){
  int nfloats = (argc>1)? atoi(argv[1]) : NFLOATS;
  int iters   = 1; (void)argc;
  int nclusters = 8; // fill the GPU
  int blocks = nclusters*CLUSTER_SIZE;
  dim3 grid(blocks), block(32);
  cudaFuncSetAttribute(dsm_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(global_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  float* gp; cudaMalloc(&gp, (size_t)nclusters*MAXF*sizeof(float));
  std::vector<unsigned long long> dsm(1024), glb(1024);
  const int REPEAT=2;
  std::vector<double> dsm_med, glb_med;
  for (int r=0; r<REPEAT; ++r){
    dsm_kernel<<<grid,block>>>(nfloats, iters);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(dsm.data(), g_dsm_cyc, 1024*sizeof(unsigned long long));
    global_kernel<<<grid,block>>>(gp, nfloats, iters);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(glb.data(), g_glb_cyc, 1024*sizeof(unsigned long long));
    // median of nonzero slots
    std::vector<unsigned long long> dv, gv;
    for (auto x: dsm) if (x>0) dv.push_back(x);
    for (auto x: glb) if (x>0) gv.push_back(x);
    if(dv.empty()||gv.empty()){ printf("ERR empty r=%d\n",r); continue; }
    std::sort(dv.begin(),dv.end()); std::sort(gv.begin(),gv.end());
    dsm_med.push_back((double)dv[dv.size()/2]/iters);
    glb_med.push_back((double)gv[gv.size()/2]/iters);
  }
  auto stat=[](std::vector<double>&v,double&m,double&sd){
    m=0; for(double x:v)m+=x; m/=v.size();
    sd=0; for(double x:v)sd+=(x-m)*(x-m); sd=sqrt(sd/v.size());
  };
  double dm,dsd,gm,gsd; stat(dsm_med,dm,dsd); stat(glb_med,gm,gsd);
  printf("nfloats=%d bytes=%d cluster=%d | DSM=%.1f±%.1f cyc/read  GLOBAL=%.1f±%.1f cyc/read  ratio(dsm/glob)=%.3f  N=%zu\n",
         nfloats, nfloats*4, CLUSTER_SIZE, dm, dsd, gm, gsd, gm>0?dm/gm:0.0, dsm_med.size());
  cudaFree(gp);
  return 0;
}
