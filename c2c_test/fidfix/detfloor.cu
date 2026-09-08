// detfloor.cu -- DETECTION-FLOOR CALIBRATION for the RESC simulator.
//
// Purpose: establish the sensitivity of the instrument used to measure
// L1-vs-L2 effects.  A null measured by an instrument that cannot detect the
// effect is not evidence.  This kernel is a DEPENDENT-READ (pointer-chase)
// microbenchmark: every load's address depends on the previous load's value,
// so there is NO memory-level parallelism WITHIN a warp and latency cannot be
// hidden by the load/store unit.  The only latency hiding available is
// thread-level parallelism (other warps), which is exactly the axis we sweep.
//
// Two arms, identical instruction stream, differing ONLY in the cache operator:
//   L1 arm : ld.global.ca  -> allocates in L1, so a re-traversal hits L1
//   L2 arm : ld.global.cg  -> bypasses L1, so every access goes to L2
// The measured per-access gap between them IS the detectable L1-vs-L2 signal.
//
// Sweep WARPS_PER_SM (via block size, one CTA per SM) and report where the
// signal survives.  Expected shape: large gap at 1-2 warps/SM decaying as TLP
// hides the latency.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime_api.h>

#ifndef WARPS
#define WARPS 1
#endif
#ifndef CHASE_KB
#define CHASE_KB 16
#endif
#ifndef STEPS
#define STEPS 512
#endif
#ifndef ARM
#define ARM 0          // 0 = .ca (L1-allocating), 1 = .cg (L1-bypassing)
#endif
#ifndef NBLOCKS
#define NBLOCKS 1
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

__device__ __forceinline__ int ld_ca_s32(const int* p){ int v;
  asm volatile("ld.global.ca.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ int ld_cg_s32(const int* p){ int v;
  asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }

__device__ unsigned long long g_cyc[64];
__device__ int                g_sink[64];

__global__ void chase(const int* __restrict__ ring, int nidx, int steps)
{
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  // Each warp walks its own independent ring so warps do not share lines;
  // adding warps adds TLP without changing the per-warp access pattern.
  const int* base = ring + (size_t)warp * (size_t)nidx;

  // warm-up traversal: brings the ring into whichever level the arm allows
  int idx = lane;
  for (int s = 0; s < nidx; ++s) {
#if ARM == 0
    idx = ld_ca_s32(&base[idx]);
#else
    idx = ld_cg_s32(&base[idx]);
#endif
  }

  __syncthreads();
  unsigned long long t0 = clock64();
  // timed dependent traversal -- strictly serial per thread
  for (int s = 0; s < steps; ++s) {
#if ARM == 0
    idx = ld_ca_s32(&base[idx]);
#else
    idx = ld_cg_s32(&base[idx]);
#endif
  }
  unsigned long long t1 = clock64();

  if (lane == 0 && blockIdx.x == 0) {
    g_cyc[warp]  = (t1 > t0) ? (t1 - t0) : 0ULL;
    g_sink[warp] = idx;
  }
}

int main(){
  const int nidx   = (CHASE_KB*1024)/(int)sizeof(int);
  const int nwarps = WARPS;
  const int nthr   = nwarps*32;
  // Build a stride-33 permutation ring per warp: each step lands on a new
  // 128 B line, so the chase is line-granular and defeats prefetch.
  const size_t total = (size_t)nidx*(size_t)nwarps;
  int* h = (int*)malloc(total*sizeof(int));
  for (int w = 0; w < nwarps; ++w)
    for (int i = 0; i < nidx; ++i)
      h[(size_t)w*nidx + i] = ( (i + 33) % nidx );
  int* d=nullptr;
  CUDA_CHECK(cudaMalloc(&d, total*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h, total*sizeof(int), cudaMemcpyHostToDevice));

  chase<<<NBLOCKS, nthr>>>(d, nidx, STEPS);
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long cyc[64]; int sink[64];
  CUDA_CHECK(cudaMemcpyFromSymbol(cyc,  g_cyc,  sizeof(cyc)));
  CUDA_CHECK(cudaMemcpyFromSymbol(sink, g_sink, sizeof(sink)));
  double sum=0; int n=0;
  for (int w=0; w<nwarps && w<64; ++w){ if(cyc[w]>0){ sum += (double)cyc[w]; ++n; } }
  double per_access = (n? sum/n : 0.0)/(double)STEPS;
  printf("DETFLOOR arm=%d warps=%d chase_kb=%d steps=%d cyc_per_dependent_access=%.3f sink=%d\n",
         ARM, WARPS, CHASE_KB, STEPS, per_access, sink[0]);
  return 0;
}
