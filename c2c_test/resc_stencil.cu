// resc_stencil.cu : cluster-cooperative 2D 5-point Jacobi stencil / halo exchange
// (W-second real-kernel end-to-end for RESC, non-tensor-core).
//
// Pattern (producer->consumer VALUE sharing with REUSE across a thread-block
// cluster -- RESC's dominant regime):
//   - The cluster's CS blocks tile a 1-D strip super-tile: block rank r owns a
//     TILE_DIM x TILE_DIM sub-tile; ranks r-1 / r+1 are its horizontal neighbors.
//   - Each Jacobi sweep, every block reads its neighbor blocks' BOUNDARY (halo)
//     column to compute its own edge cells. The halo cells are re-read every
//     iteration (= reuse), and each is one producer's freshly written value
//     (= producer->consumer). Interior cells stay within the block's own tile.
//   - Double-buffered (cur/next) real Jacobi average -> pure float add/mul,
//     NON-tensor-core, so it avoids the simulator FP-tensor fidelity gap.
//
// BACKEND (structurally identical to resc_md.cu):
//   0 = GLOBAL : halo exchanged through global memory with ld.cg (bypasses L1,
//                refetches every iter). Correct-legal baseline.
//   1 = DSM    : each block stages its current tile into SMEM; neighbor halo
//                read via cluster.map_shared_rank (reserves SMEM -> occupancy cost).
//   2 = RESC   : own tile marked cluster-coherent (register_range); neighbor
//                halo served L1-to-L1 across iters (no SMEM reserve).
//   4 = noncoh : unsafe L1 caching of halo (ld.ca), no coherence -- upper bound.
//
// Build (CUDA 12.4 nvcc so it links the sim's cudart.so.12 shim):
//   nvcc -arch=sm_90 -DBACKEND=0 -DCLUSTER_SIZE=4 -DTILE_DIM=32 resc_stencil.cu -o st_cg   -lcudart
//   nvcc -arch=sm_90 -DBACKEND=1 -DCLUSTER_SIZE=4 -DTILE_DIM=32 resc_stencil.cu -o st_dsm  -lcudart
//   nvcc -arch=sm_90 -DBACKEND=2 -DCLUSTER_SIZE=4 -DTILE_DIM=32 -DREAD_MODE=1 -DWRITE_MODE=1 resc_stencil.cu -o st_resc -lcudart
//   nvcc -arch=sm_90 -DBACKEND=4 -DCLUSTER_SIZE=4 -DTILE_DIM=32 resc_stencil.cu -o st_noncoh -lcudart
//   then: patchelf --remove-rpath <out>
//
// Run:
//   ./st_resc num_clusters=8 iters=32
//   args: num_clusters=<n>  iters=<n>   (cluster size + tile dim are compile-time)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;

#ifndef BACKEND
#define BACKEND 0
#endif

#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 4    // blocks per cluster (compile-time; pass -DCLUSTER_SIZE)
#endif

#ifndef TILE_DIM
#define TILE_DIM 32       // sub-tile edge in cells; row = TILE_DIM floats
#endif

#define TILE_ELEMS (TILE_DIM * TILE_DIM)

#ifndef RESC_LINE_BYTES
#define RESC_LINE_BYTES 128
#endif

#ifndef READ_MODE
#define READ_MODE 0   // BACKEND=2: 0=ld.cg, 1=ld.ca
#endif
#ifndef WRITE_MODE
#define WRITE_MODE 0  // BACKEND=2: 0=st.cg, 1=st.wb
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// ---- PTX helpers (identical to resc_md.cu) ----
__device__ __forceinline__ float ld_cg_f32(const float* p){ float v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.cg.f32 %0,[%1];" : "=f"(v) : "l"(p));
#else
  v=*p;
#endif
  return v;
}
__device__ __forceinline__ void st_cg_f32(float* p,float v){
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.cg.f32 [%0],%1;" :: "l"(p),"f"(v) : "memory");
#else
  *p=v;
#endif
}
__device__ __forceinline__ float ld_ca_f32(const float* p){ float v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p));
#else
  v=*p;
#endif
  return v;
}
__device__ __forceinline__ void st_wb_f32(float* p,float v){
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.wb.f32 [%0],%1;" :: "l"(p),"f"(v) : "memory");
#else
  *p=v;
#endif
}

// simulator hook (identical to resc_md.cu)
__device__ volatile int g_dummy_registration_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_dummy_registration_flag = (int)((unsigned long long)addr & 0x1);
  (void)addr;
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)RESC_LINE_BYTES - 1);
}
__device__ __forceinline__ void register_range_by_line_warp0(void* base, size_t bytes){
  uintptr_t b = (uintptr_t)base;
  uintptr_t start = align_down_line(b);
  int nlines = (int)(((b + bytes - start) + RESC_LINE_BYTES - 1) / RESC_LINE_BYTES);
  int lane = threadIdx.x & 31;
  for(int off=0; off<nlines; off+=32){
    int idx = off + lane;
    uintptr_t line = start + (uintptr_t)((idx < nlines) ? idx : 0) * RESC_LINE_BYTES;
    gpgpusim_register_cluster_coherent_addr((void*)line);
  }
}

// ---- read/write selection by BACKEND (mirrors resc_md.cu) ----
#if BACKEND == 2
  #define READ_F32(p)   (READ_MODE==1 ? ld_ca_f32(p) : ld_cg_f32(p))
  #define WRITE_F32(p,v) do{ if(WRITE_MODE==1) st_wb_f32(p,v); else st_cg_f32(p,v);}while(0)
#elif BACKEND == 0
  #define READ_F32(p)   ld_cg_f32(p)
  #define WRITE_F32(p,v) st_cg_f32(p,v)
#elif BACKEND == 4
  // Non-coherent L1-residency CONTROL: read halo with ld.ca (L1 caching), no
  // RESC protocol. Isolates bare-L1-residency benefit from the coherence
  // machinery. Safe here only by construction (single producer per boundary
  // column, published before consumers read within a synced epoch).
  #define READ_F32(p)   ld_ca_f32(p)
  #define WRITE_F32(p,v) st_cg_f32(p,v)
#endif

// Halo (neighbor) read: DSM = plain mapped-SMEM access; others = READ_F32.
#if BACKEND == 1
__device__ __forceinline__ float read_halo(const float* base, int idx){ return base[idx]; }
#else
__device__ __forceinline__ float read_halo(const float* base, int idx){ return READ_F32(base + idx); }
#endif

// ============================================================================
// stencil_step: one kernel does `iters` Jacobi sweeps over the cluster strip.
// cluster = CS blocks; block rank r owns cell (cluster_id*CS + r).
// Halo = neighbor rank's boundary column, read every sweep (reuse).
// Single-producer-per-line: each cell's tile has exactly one producer (owner);
// neighbors only READ the boundary column -> preserves single-producer safety.
// ============================================================================
template<int CS>
__global__ __cluster_dims__(CS,1,1)
void stencil_step(float* __restrict__ g_bufA,   // [num_cells * TILE_ELEMS]
                  float* __restrict__ g_bufB,
                  float* __restrict__ g_out,     // per-cell final checksum sink
                  int num_clusters, int iters)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid  = threadIdx.x;
  const int cluster_id = (int)(blockIdx.x / CS);
  const int cell = cluster_id * CS + rank;      // global cell owned by this block

#if BACKEND == 1
  __shared__ float smem_tile[TILE_ELEMS];       // DSM: stage own tile so neighbors can map it
#endif

  // RESC: mark this cell's tile (in BOTH ping-pong buffers) cluster-coherent once.
#if BACKEND == 2
  if (tid < 32) {
    register_range_by_line_warp0(g_bufA + (size_t)cell*TILE_ELEMS, (size_t)TILE_ELEMS*sizeof(float));
    register_range_by_line_warp0(g_bufB + (size_t)cell*TILE_ELEMS, (size_t)TILE_ELEMS*sizeof(float));
  }
  __syncwarp();
#endif

  float* g_cur  = g_bufA;
  float* g_next = g_bufB;
  cluster.sync();

  for (int it = 0; it < iters; ++it) {
    float* mine = g_cur + (size_t)cell*TILE_ELEMS;

#if BACKEND == 1
    // DSM: stage current tile into SMEM so peers can map_shared_rank it.
    for (int k = tid; k < TILE_ELEMS; k += blockDim.x) smem_tile[k] = mine[k];
    __syncthreads();
#endif
    cluster.sync();   // epoch barrier: all neighbors' current tiles ready before halo reads

    // neighbor tile bases for the halo reads
    const bool has_left  = (rank > 0);
    const bool has_right = (rank < CS - 1);
#if BACKEND == 1
    // map_shared_rank must ALWAYS receive a valid rank (0..CS-1): the sim's mapa
    // handler computes the address regardless of a C-level predicate guard, so a
    // clamped rank is passed at edges and the mapped pointer is simply not read
    // (has_left/has_right gates the actual halo read below). Mirrors resc_md.cu's
    // invariant that map_shared_rank never sees an out-of-range rank.
    int lrank = has_left  ? (rank-1) : rank;
    int rrank = has_right ? (rank+1) : rank;
    const float* lbase = (const float*)cluster.map_shared_rank(smem_tile, lrank);
    const float* rbase = (const float*)cluster.map_shared_rank(smem_tile, rrank);
#else
    const float* lbase = has_left  ? (g_cur + (size_t)(cell-1)*TILE_ELEMS) : nullptr;
    const float* rbase = has_right ? (g_cur + (size_t)(cell+1)*TILE_ELEMS) : nullptr;
#endif

    // ---- Jacobi sweep: compute next from cur (5-point average) ----
    for (int idx = tid; idx < TILE_ELEMS; idx += blockDim.x) {
      int i = idx / TILE_DIM;   // row
      int j = idx % TILE_DIM;   // col
      float c    = mine[i*TILE_DIM + j];
      float up   = (i > 0)          ? mine[(i-1)*TILE_DIM + j] : c;             // Neumann top/bottom
      float down = (i < TILE_DIM-1) ? mine[(i+1)*TILE_DIM + j] : c;
      float left, right;
      if (j > 0)              left  = mine[i*TILE_DIM + (j-1)];
      else                    left  = has_left  ? read_halo(lbase, i*TILE_DIM + (TILE_DIM-1)) : c;  // HALO
      if (j < TILE_DIM-1)     right = mine[i*TILE_DIM + (j+1)];
      else                    right = has_right ? read_halo(rbase, i*TILE_DIM + 0)            : c;  // HALO
      float nv = 0.2f * (c + up + down + left + right);
      g_next[(size_t)cell*TILE_ELEMS + i*TILE_DIM + j] = nv;
    }

    cluster.sync();   // all next[] written before the swap makes it the new cur
    float* t = g_cur; g_cur = g_next; g_next = t;
  }

  // ---- checksum of this cell's final tile (identical across arms 0/1/2) ----
  float* fin = g_cur + (size_t)cell*TILE_ELEMS;
  float local = 0.f;
  for (int idx = tid; idx < TILE_ELEMS; idx += blockDim.x) local += fin[idx];
  // block reduction into g_out[cell]
  __shared__ float red[256];
  red[tid] = local; __syncthreads();
  for (int s = blockDim.x/2; s > 0; s >>= 1){ if (tid < s) red[tid]+=red[tid+s]; __syncthreads(); }
  if (tid==0) g_out[cell] = red[0];
#endif
}

int main(int argc, char** argv){
  int cluster_size = CLUSTER_SIZE;
  int num_clusters = 8;
  int iters = 32;
  for (int i=1;i<argc;i++){
    if(!strncmp(argv[i],"num_clusters=",13)) num_clusters=atoi(argv[i]+13);
    else if(!strncmp(argv[i],"iters=",6)) iters=atoi(argv[i]+6);
  }
  int num_cells = num_clusters * cluster_size;
  size_t nelem = (size_t)num_cells * TILE_ELEMS;

  float *d_A,*d_B,*d_out;
  CUDA_CHECK(cudaMalloc(&d_A, nelem*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, nelem*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, num_cells*sizeof(float)));

  float* h_A=(float*)malloc(nelem*sizeof(float));
  for(int c=0;c<num_cells;c++){
    for(int idx=0; idx<TILE_ELEMS; idx++){
      unsigned s=(unsigned)((c*2654435761u) ^ (idx*40503u)); s^=s>>13;
      h_A[(size_t)c*TILE_ELEMS+idx] = (float)(s % 1000) * 0.01f;   // in [0,10)
    }
  }
  CUDA_CHECK(cudaMemcpy(d_A,h_A,nelem*sizeof(float),cudaMemcpyHostToDevice));
  // seed B = A so a 0-iter run and Neumann edges are well-defined/identical
  CUDA_CHECK(cudaMemcpy(d_B,h_A,nelem*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out,0,num_cells*sizeof(float)));

  dim3 grid(cluster_size*num_clusters,1,1); dim3 block(128,1,1);
  stencil_step<CLUSTER_SIZE><<<grid,block>>>(d_A,d_B,d_out,num_clusters,iters);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

  float* h_out=(float*)malloc(num_cells*sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_out,d_out,num_cells*sizeof(float),cudaMemcpyDeviceToHost));
  double sum=0; for(int c=0;c<num_cells;c++) sum+=h_out[c];
  printf("OUT_CHECKSUM sum=%.6f n=%d BACKEND=%d cluster=%d tile=%d num_clusters=%d iters=%d\n",
         sum,num_cells,BACKEND,cluster_size,TILE_DIM,num_clusters,iters);
  return 0;
}
