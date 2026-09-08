// selprobe.cu -- SELECTIVITY POSITIVE CONTROL.
//
// Pre-registration: selective-invalidate-prereg.md S5, SHA-256 8ebabe19...,
// frozen 2026-09-03T11:11:43Z (BEFORE this file was written).
//
// Question: does knob 2 (marked-line selective invalidate) actually PRESERVE
// unmarked lines while STILL invalidating marked ones?  This is the case where
// selective and whole-L1 invalidate MUST differ.  If they do not differ here,
// the implementation is wrong.
//
// Construction: two disjoint working sets touched by the SAME instruction
// (ld.global.ca), so the ONLY difference between them is markedness.
//   * UNMARKED: a per-block PRIVATE tile (TILE_KB), never registered, never
//     communicated, re-read once per epoch.  A whole-L1 invalidate destroys it
//     every epoch (one compulsory pass per epoch); a marked-line-selective
//     invalidate must leave it alone.
//   * MARKED: a small per-cluster communication buffer (COMM_KB), registered
//     with gpgpusim_register_cluster_coherent_addr, written by rank 0 and read
//     by every rank after the barrier.  Both knob 1 and knob 2 must invalidate
//     it (knob 2 is not allowed to be a no-op).
//
// Required signature (prereg S5 / S9.3):
//   knob 0: unmarked misses ~ compulsory only   marked misses low
//   knob 1: unmarked misses ~ 32x compulsory    marked misses high
//   knob 2: unmarked misses ~ compulsory ONLY   marked misses high (~knob 1)
//
// knob 2 collapsing to knob 0 => the marked test never fires.
// knob 2 collapsing to knob 1 => the unmarked preservation never fires.
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef EPOCHS
#define EPOCHS 32
#endif
#ifndef TILE_KB
#define TILE_KB 16          // per-block PRIVATE UNMARKED tile
#endif
#ifndef COMM_KB
#define COMM_KB 1           // per-cluster MARKED communication buffer
#endif
#ifndef CS
#define CS 2
#endif
#ifndef NCLUSTERS
#define NCLUSTERS 2
#endif
#define LINE_BYTES 128

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// simulator hook: intercepted in cuda-sim/instructions.cc:2761
__device__ volatile int g_reg_flag = 0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_reg_flag = (int)((unsigned long long)addr & 1); (void)addr;
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)LINE_BYTES - 1);
}
__device__ __forceinline__ void register_range_warp0(void* base, size_t bytes){
  uintptr_t b=(uintptr_t)base, start=align_down_line(b);
  int nlines=(int)(((b+bytes-start)+LINE_BYTES-1)/LINE_BYTES);
  int lane=threadIdx.x&31;
  for(int off=0;off<nlines;off+=32){
    int idx=off+lane;
    uintptr_t line=start+(uintptr_t)((idx<nlines)?idx:0)*LINE_BYTES;
    gpgpusim_register_cluster_coherent_addr((void*)line);
  }
}
__device__ __forceinline__ float ld_ca_f32(const float* p){ float v;
  asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_wb_f32(float* p, float v){
  asm volatile("st.global.wb.f32 [%0],%1;" :: "l"(p),"f"(v) : "memory"); }

__global__ void __cluster_dims__(CS,1,1)
selprobe(float* __restrict__ tile, float* __restrict__ comm,
         float* __restrict__ out, int tile_elems, int comm_elems)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cl = cg::this_cluster();
  const int rank = cl.block_rank();
  const int tid  = threadIdx.x;
  const int blk  = blockIdx.x;
  const int cid  = blk / CS;

  // UNMARKED: private per-block tile.  Deliberately NOT registered.
  float* mytile = tile + (size_t)blk * (size_t)tile_elems;
  // MARKED: per-cluster communication buffer.
  float* mycomm = comm + (size_t)cid * (size_t)comm_elems;

  // Register ONLY the communication buffer -- this is what makes it "marked".
  if (rank == 0 && tid < 32)
    register_range_warp0(mycomm, (size_t)comm_elems * sizeof(float));
  __syncthreads();
  cl.sync();

  float acc_u = 0.f, acc_m = 0.f;
  for (int e = 0; e < EPOCHS; ++e) {
    // ---- UNMARKED traffic: one pass over the private tile, every epoch.
    for (int i = tid; i < tile_elems; i += blockDim.x)
      acc_u += ld_ca_f32(&mytile[i]);

    // ---- MARKED traffic: rank 0 produces this epoch's payload.
    if (rank == 0)
      for (int i = tid; i < comm_elems; i += blockDim.x)
        st_wb_f32(&mycomm[i], (float)(e * 131 + i));

    cl.sync();   // THE ACQUIRE under test

    // ---- MARKED traffic: every rank consumes it after the acquire.
    for (int i = tid; i < comm_elems; i += blockDim.x)
      acc_m += ld_ca_f32(&mycomm[i]);
  }
  if (tid == 0) out[blk] = acc_u + acc_m;
#endif
}

int main(){
  const int tile_elems = (TILE_KB*1024)/(int)sizeof(float);
  const int comm_elems = (COMM_KB*1024)/(int)sizeof(float);
  const int nblocks    = CS*NCLUSTERS;
  float *d_tile=nullptr,*d_comm=nullptr,*d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_tile,(size_t)nblocks*tile_elems*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_comm,(size_t)NCLUSTERS*comm_elems*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out,(size_t)nblocks*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_tile,0,(size_t)nblocks*tile_elems*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_comm,0,(size_t)NCLUSTERS*comm_elems*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out,0,(size_t)nblocks*sizeof(float)));
  selprobe<<<nblocks,256>>>(d_tile,d_comm,d_out,tile_elems,comm_elems);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float* h=(float*)malloc((size_t)nblocks*sizeof(float));
  CUDA_CHECK(cudaMemcpy(h,d_out,(size_t)nblocks*sizeof(float),cudaMemcpyDeviceToHost));
  double s=0; for(int i=0;i<nblocks;++i) s+=h[i];
  const long long unmarked_lines = (long long)nblocks*(tile_elems*4/LINE_BYTES);
  const long long marked_lines   = (long long)NCLUSTERS*(comm_elems*4/LINE_BYTES);
  std::printf("SELPROBE EPOCHS=%d TILE_KB=%d COMM_KB=%d CS=%d NCLUSTERS=%d "
              "nblocks=%d tile_elems=%d comm_elems=%d "
              "unmarked_lines=%lld marked_lines=%lld "
              "expect_unmarked_miss_knob0=%lld expect_unmarked_miss_knob1=%lld "
              "checksum=%.6f\n",
              EPOCHS,TILE_KB,COMM_KB,CS,NCLUSTERS,nblocks,tile_elems,comm_elems,
              unmarked_lines, marked_lines,
              unmarked_lines, unmarked_lines*(long long)EPOCHS, s);
  free(h);
  return 0;
}
