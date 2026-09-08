// RESC cross-cluster (inter-GPC) producer/consumer microbenchmark -- CCOP arms.
//
// Fixes vs the original (per CCOP-design-brief):
//   (1) REUSE: each consumer re-reads the SAME producer regions R times. The
//       strip working set is sized ABOVE the L1 (32KB) so every pass re-misses
//       the consumer L1 -> read#1 trains CCOP via L2, reads#2..R take the
//       directed C2C probe (this is what the predictor is for).
//   (2) CROSS-GPC + 8KB region layout: each producer owns one 8KB-aligned slice
//       (=> a distinct CCOP region entry, stable owner), consumers are issued
//       after all producers so they land on DIFFERENT GPCs. The sim prints
//       crossgpc_read_frac; the runner asserts it > 0.9 else INVALID.
//
// Arms are selected sim-side by env RESC_C2C_MODE (0=A baseline, 1=S1 CCOP
// default, 2=S2 probe-first) and RESC_C2C_NOPROBE=1 (C cold-only control:
// CCOP present but never probes -> must equal A, zero regression).
//
// Metric: gpu_tot_sim_cycle + [C2C_STATS]/[CCOP_STATS] counters + OUT_CHECKSUM
// (must match across all arms -- functional correctness proof).
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#ifndef NREG
#define NREG 32            // number of producer regions (<=128: one CCOP slot each)
#endif
#ifndef RFLOATS
#define RFLOATS 512        // floats per region payload (512*4=2KB = 16 L1 lines)
#endif
#ifndef RREUSE
#define RREUSE 64          // reuse passes per consumer over the strip
#endif
#ifndef NCONS
#define NCONS 4            // number of consumer blocks (each reads the full strip)
#endif

// 8KB-aligned producer slice so region_key = addr>>13 is unique per producer.
#define REGION_STRIDE_FLOATS 2048   // 2048*4 = 8192 B = 8KB

// Sim hook: mark a global address as a cross-cluster communication line.
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cross_cluster_addr(void* addr){
  asm volatile("" :: "l"(addr) : "memory");   // intercepted functionally by sim
}

__device__ unsigned long long g_checksum = 0;
__device__ int g_cons_done = 0;               // consumers finished (producer keep-alive)

// blockIdx.x layout: [0, nreg)                 -> producers (region = bid)
//                    [nreg, nreg+ncons)        -> consumers (each reads full strip)
__global__ void pc_kernel(float* payload, volatile int* ready,
                          int nreg, int rfloats, int rreuse, int ncons){
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (bid < nreg) {
    // ---------------- PRODUCER for region = bid ----------------
    int r = bid;
    float* base = payload + (size_t)r * REGION_STRIDE_FLOATS;
    // mark region lines cross-cluster (warp0 lane-strided over lines)
    if (tid < 32) {
      for (int i = tid; i < rfloats; i += 32)
        gpgpusim_register_cross_cluster_addr((void*)&base[i]);
    }
    __syncthreads();
    // producer store (single writer). Functional value is authoritative in the
    // functional sim; the WT invariant is what makes an owner serve legitimate.
    for (int i = tid; i < rfloats; i += blockDim.x) {
      float v = (float)(r * 1000 + i);
#if __CUDA_ARCH__ >= 900
      asm volatile("st.global.wb.f32 [%0],%1;" :: "l"(&base[i]), "f"(v) : "memory");
#else
      base[i] = v;
#endif
    }
    __threadfence();
    // read-back so the produced lines are resident+readable in the producer L1.
    float rb = 0.f;
    for (int i = tid; i < rfloats; i += blockDim.x) {
      float v; asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
      rb += v;
    }
    if (rb == -123456.f) payload[0] = rb;   // prevent DCE
    __syncthreads();
    if (tid == 0) ready[r] = 1;
    // Keep-alive: producer stays resident and its SM stays busy (so consumers
    // cannot be scheduled onto producer GPCs) until all consumers finish. Also
    // models a producer still actively using its data (temporal overlap).
    for (int s = 0; s < 2000000 && g_cons_done < ncons; ++s) {
      float hot = 0.f;
      for (int i = tid; i < rfloats; i += blockDim.x) {
        float v; asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
        hot += v;
      }
      if (hot == -987654.f) payload[0] = hot;
      __threadfence();
    }
  } else {
    // ---------------- CONSUMER (reads the whole strip R times) ----------------
    int cid = bid - nreg;
    if (cid >= ncons) return;
    // wait until every producer has published
    if (tid == 0) { for (int r = 0; r < nreg; ++r) while (ready[r] == 0) __threadfence(); }
    __syncthreads();

    float acc = 0.f;
    // REUSE: R passes over all nreg regions. Strip footprint =
    //   nreg * (rfloats/32) lines. With nreg=32, rfloats=512 -> 512 lines =
    //   64KB > 32KB L1 -> each pass re-misses -> pass#1 trains, #2..R probe.
    for (int p = 0; p < rreuse; ++p) {
      for (int r = 0; r < nreg; ++r) {
        float* base = payload + (size_t)r * REGION_STRIDE_FLOATS;
        for (int i = tid; i < rfloats; i += blockDim.x) {
          float v;
#if __CUDA_ARCH__ >= 900
          asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(&base[i]) : "memory");
#else
          v = base[i];
#endif
          acc += v;
        }
      }
    }
    atomicAdd(&g_checksum, (unsigned long long)(long long)acc);
    __syncthreads();
    if (tid == 0) atomicAdd(&g_cons_done, 1);   // release producers
  }
}

int main(int argc, char** argv){
  int nreg    = (argc>1)? atoi(argv[1]) : NREG;
  int rfloats = (argc>2)? atoi(argv[2]) : RFLOATS;
  int rreuse  = (argc>3)? atoi(argv[3]) : RREUSE;
  int ncons   = (argc>4)? atoi(argv[4]) : NCONS;
  int nblocks = nreg + ncons;
  dim3 grid(nblocks), block(32);

  size_t nfloats = (size_t)nreg * REGION_STRIDE_FLOATS;   // 8KB-strided buffer
  float* payload; int* ready;
  cudaMalloc(&payload, nfloats*sizeof(float));
  cudaMalloc(&ready, (size_t)nreg*sizeof(int));
  cudaMemset(ready, 0, (size_t)nreg*sizeof(int));
  cudaMemset(payload, 0, nfloats*sizeof(float));

  pc_kernel<<<grid, block>>>(payload, ready, nreg, rfloats, rreuse, ncons);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CUDA_ERR=%s\n", cudaGetErrorString(e)); return 1; }

  unsigned long long cs = 0;
  cudaMemcpyFromSymbol(&cs, g_checksum, sizeof(cs));
  printf("OUT_CHECKSUM sum=%llu nreg=%d rfloats=%d rreuse=%d ncons=%d blocks=%d "
         "strip_lines=%d strip_KB=%d\n",
         cs, nreg, rfloats, rreuse, ncons, nblocks,
         nreg*(rfloats/32), (nreg*rfloats*4)/1024);
  cudaFree(payload); cudaFree(ready);
  return 0;
}
