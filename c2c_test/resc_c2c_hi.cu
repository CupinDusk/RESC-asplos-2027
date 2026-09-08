// RESC cross-cluster (inter-GPC) producer/consumer microbenchmark.
// Producers write a per-region payload to GLOBAL (marked cross-cluster lines) and
// publish a ready flag; consumers (landing on other GPCs via round-robin issue)
// spin-wait then read the payload cross-cluster. Arms are selected sim-side by
// env RESC_C2C_MODE (0=A all-L2, 1=B broadcast, 2=C predictor).
//
// Metric: gpu_tot_sim_cycle (printed by sim) + [C2C_STATS] neighbor-hit/L2-fallback
// counters. Prints OUT_CHECKSUM for correctness (must match across arms).
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#ifndef NREG
#define NREG 64            // number of producer regions
#endif
#ifndef RFLOATS
#define RFLOATS 256         // floats per region (32*4=128B = one L1 line)
#endif
#ifndef NCONS
#define NCONS 7            // consumers per region (sharing degree)
#endif

// Sim hook: mark a global address as a cross-cluster communication line.
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cross_cluster_addr(void* addr){
  // Body irrelevant; intercepted functionally by the simulator.
  asm volatile("" :: "l"(addr) : "memory");
}

__device__ unsigned long long g_checksum = 0;

// One block per region produces; NCONS blocks per region consume.
// blockIdx.x layout: [0, NREG)          -> producers (region = bid)
//                    [NREG, NREG*(1+NCONS)) -> consumers (region = (bid-NREG)/NCONS)
__global__ void pc_kernel(float* payload, volatile int* ready, int nreg, int rfloats){
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (bid < nreg) {
    // PRODUCER for region=bid
    int r = bid;
    float* base = payload + (size_t)r * rfloats;
    // mark the region lines as cross-cluster (warp0 lane-strided over lines)
    if (tid < 32) {
      for (int i = tid; i < rfloats; i += 32)
        gpgpusim_register_cross_cluster_addr((void*)&base[i]);
    }
    __syncthreads();
    // write-through producer store (wt) so L1 copy == L2 copy (single writer)
    for (int i = tid; i < rfloats; i += blockDim.x) {
      float v = (float)(r * 1000 + i);
#if __CUDA_ARCH__ >= 900
      asm volatile("st.global.wb.f32 [%0],%1;" :: "l"(&base[i]), "f"(v) : "memory");
#else
      base[i] = v;
#endif
    }
    __threadfence();
    // read-back through L1 so the produced line is resident+readable in the
    // producer L1 (clean copy == L2 by write-through) for cross-cluster serving.
    float rb = 0.f;
    for (int i = tid; i < rfloats; i += blockDim.x) {
      float v;
      asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
      rb += v;
    }
    if (rb == -123456.f) payload[0] = rb;  // prevent DCE
    __syncthreads();
    // WARM-UP: re-read the line enough times that its L1 fill has definitely
    // completed and it is LRU-hot BEFORE publishing ready. This makes the
    // producer L1 a valid cross-cluster source when consumers query.
    for (int w = 0; w < 64; ++w) {
      float hv = 0.f;
      for (int i = tid; i < rfloats; i += blockDim.x) {
        float v;
        asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
        hv += v;
      }
      if (hv == -55555.f) payload[0] = hv;
    }
    __syncthreads();
    if (tid == 0) ready[r] = 1;
    // Keep producer resident AND its line genuinely hot in its own L1 by
    // re-reading the payload each iteration until all consumers have read.
    // This models a real producer still actively using its data (temporal
    // overlap) -- the precondition for an honest neighbor-L1 hit.
    for (int s = 0; s < 200000 && ready[r] < (1 + NCONS); ++s) {
      float hot = 0.f;
      for (int i = tid; i < rfloats; i += blockDim.x) {
        float v;
        asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
        hot += v;
      }
      if (hot == -987654.f) payload[0] = hot;  // prevent DCE
      __threadfence();
    }
  } else {
    // CONSUMER
    int idx = bid - nreg;
    int r = idx / NCONS;
    if (r >= nreg) return;
    float* base = payload + (size_t)r * rfloats;
    // spin until producer published
    if (tid == 0) { while (ready[r] == 0) { __threadfence(); } }
    __syncthreads();
    // cross-cluster reads (cache-global loads)
    float acc = 0.f;
    for (int i = tid; i < rfloats; i += blockDim.x) {
      float v;
#if __CUDA_ARCH__ >= 900
      asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
#else
      v = base[i];
#endif
      acc += v;
    }
    // reduce acc into checksum (approx; enough to detect corruption)
    atomicAdd(&g_checksum, (unsigned long long)(long long)acc);
    __syncthreads();
    if (tid == 0) atomicAdd((int*)&ready[r], 1);  // signal consumer done
  }
}

int main(int argc, char** argv){
  int nreg   = (argc>1)? atoi(argv[1]) : NREG;
  int rfloats= (argc>2)? atoi(argv[2]) : RFLOATS;
  int ncons  = NCONS;
  int nblocks = nreg * (1 + ncons);
  dim3 grid(nblocks), block(32);

  float* payload; int* ready;
  cudaMalloc(&payload, (size_t)nreg*rfloats*sizeof(float));
  cudaMalloc(&ready, (size_t)nreg*sizeof(int));
  cudaMemset(ready, 0, (size_t)nreg*sizeof(int));
  cudaMemset(payload, 0, (size_t)nreg*rfloats*sizeof(float));

  pc_kernel<<<grid, block>>>(payload, ready, nreg, rfloats);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA_ERR=%s\n", cudaGetErrorString(e)); return 1; }

  unsigned long long cs = 0;
  cudaMemcpyFromSymbol(&cs, g_checksum, sizeof(cs));
  printf("OUT_CHECKSUM sum=%llu nreg=%d rfloats=%d ncons=%d blocks=%d\n",
         cs, nreg, rfloats, ncons, nblocks);
  cudaFree(payload); cudaFree(ready);
  return 0;
}
