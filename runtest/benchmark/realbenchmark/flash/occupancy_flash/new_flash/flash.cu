// flash_modified.cu
//
// Multi-cluster + (RESERVE vs ACTUAL) version based on your latest flash.cu.
//
// BACKEND:
//   0 = GLOBAL (.cg)
//   1 = DSMEM  (cluster.map_shared_rank to rank0's SMEM)
//   2 = RESC   (register coherent addrs + selectable ld.ca/ld.cg, st.wb/st.cg)
//
// Run:
//   ./flash <cluster_size:2|4|8> <repeat> <num_clusters> <actual_elems>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;

#ifndef BACKEND
#define BACKEND 0
#endif

#ifndef RESERVE_ELEMS
#define RESERVE_ELEMS 4096
#endif

#ifndef STRIDE_ELEMS
// Keep STRIDE_ELEMS fixed across RESERVE_ELEMS sweeps to avoid footprint/cache confounds.
// Must satisfy STRIDE_ELEMS >= actual_elems.
#define STRIDE_ELEMS 16384
#endif

#ifndef RESC_LINE_BYTES
#define RESC_LINE_BYTES 128
#endif

// For RESC (and generally for global polling), avoid false sharing on flags.
// RESC coherence tracking in the simulator is cacheline-granular (block_addr),
// so packing flags as int[num_clusters] causes many clusters to share one line.
// We pad flags so each cluster gets its own cacheline.
#ifndef FLAG_STRIDE_INTS
#define FLAG_STRIDE_INTS (RESC_LINE_BYTES / (int)sizeof(int))
#endif

// Enforce cacheline isolation across clusters for RESC-registered regions.
// - Each cluster's tile slice is STRIDE_ELEMS floats; keep it cacheline-aligned.
// - Each cluster's flag occupies an entire cacheline (padding) to avoid false sharing.
static_assert((STRIDE_ELEMS * (int)sizeof(float)) % RESC_LINE_BYTES == 0,
              "STRIDE_ELEMS*sizeof(float) must be a multiple of RESC_LINE_BYTES");
static_assert((FLAG_STRIDE_INTS * (int)sizeof(int)) == RESC_LINE_BYTES,
              "FLAG_STRIDE_INTS*sizeof(int) must equal RESC_LINE_BYTES");

#ifndef READ_MODE
#define READ_MODE 0  // BACKEND=2: 0=ld.cg, 1=ld.ca
#endif
#ifndef WRITE_MODE
#define WRITE_MODE 0 // BACKEND=2: 0=st.cg, 1=st.wb
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

// ---- PTX helpers (matching your style) ----
__device__ __forceinline__ int ld_cg_s32(int* p){ int v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(v) : "l"(p));
#else
  v=*p;
#endif
  return v;
}
__device__ __forceinline__ void st_cg_s32(int* p,int v){
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.cg.s32 [%0],%1;" :: "l"(p),"r"(v) : "memory");
#else
  *p=v;
#endif
}
__device__ __forceinline__ float ld_cg_f32(float* p){ float v;
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

__device__ __forceinline__ int ld_ca_s32(int* p){ int v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.ca.s32 %0,[%1];" : "=r"(v) : "l"(p));
#else
  v=*p;
#endif
  return v;
}
__device__ __forceinline__ float ld_ca_f32(float* p){ float v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p));
#else
  v=*p;
#endif
  return v;
}
__device__ __forceinline__ void st_wb_s32(int* p,int v){
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.wb.s32 [%0],%1;" :: "l"(p),"r"(v) : "memory");
#else
  *p=v;
#endif
}
__device__ __forceinline__ void st_wb_f32(float* p,float v){
#if __CUDA_ARCH__ >= 900
  asm volatile("st.global.wb.f32 [%0],%1;" :: "l"(p),"f"(v) : "memory");
#else
  *p=v;
#endif
}

// simulator hook
__device__ volatile int g_dummy_registration_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_dummy_registration_flag = (int)((unsigned long long)addr & 0x1);
  (void)addr;
}

__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)RESC_LINE_BYTES - 1);
}

// Register [base, base+bytes) by cacheline, using warp0 of rank0 block.
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

#if BACKEND == 2
  #define READ_S32(p)  (READ_MODE==1 ? ld_ca_s32(p) : ld_cg_s32(p))
  #define READ_F32(p)  (READ_MODE==1 ? ld_ca_f32(p) : ld_cg_f32(p))
  #define WRITE_S32(p,v) do{ if(WRITE_MODE==1) st_wb_s32(p,v); else st_cg_s32(p,v);}while(0)
  #define WRITE_F32(p,v) do{ if(WRITE_MODE==1) st_wb_f32(p,v); else st_cg_f32(p,v);}while(0)
#elif BACKEND == 0
  #define READ_S32(p)  ld_cg_s32(p)
  #define READ_F32(p)  ld_cg_f32(p)
  #define WRITE_S32(p,v) st_cg_s32(p,v)
  #define WRITE_F32(p,v) st_cg_f32(p,v)
#endif

template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void fa_tile_many_clusters(float* __restrict__ g_tiles,
                           int*   __restrict__ g_flags,
                           float* __restrict__ out,
                           int repeat,
                           int actual_elems)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  int tid  = threadIdx.x;

  // Each contiguous CLUSTER_SIZE blocks form a cluster.
  int cluster_id = (int)(blockIdx.x / CLUSTER_SIZE);

  // Clamp runtime actual_elems.
  if (actual_elems > RESERVE_ELEMS) actual_elems = RESERVE_ELEMS;
  if (actual_elems > STRIDE_ELEMS)  actual_elems = STRIDE_ELEMS;
  if (actual_elems < 1) actual_elems = 1;

  float* tile_ptr = nullptr;
  int*   flag_ptr = nullptr;

#if BACKEND == 1
  // DSMEM: reserve compile-time RESERVE_ELEMS in SMEM (this is the pain point we sweep).
  __shared__ float smem_tile[RESERVE_ELEMS];
  __shared__ int   smem_flag;

  tile_ptr = (float*)cluster.map_shared_rank(smem_tile, 0);
  flag_ptr = (int*)  cluster.map_shared_rank(&smem_flag, 0);

  if (tid==0 && rank==0) atomicExch(flag_ptr, 0);
  cluster.sync();
#else
  // GLOBAL/RESC: per-cluster slice + per-cluster flag (avoid cross-cluster interference).
  tile_ptr = g_tiles + (size_t)cluster_id * (size_t)STRIDE_ELEMS;
  flag_ptr = g_flags + (size_t)cluster_id * (size_t)FLAG_STRIDE_INTS;

  if (tid==0 && rank==0) WRITE_S32(flag_ptr, 0);

  // RESC: register only the ACTUAL bytes accessed (keep registration overhead constant across RESERVE sweeps).
  // NOTE: Simulator registration is now per (gpc_id, cluster_slot) on the host side,
  // but coherence/sharing is still cacheline-granular. Keep per-cluster regions
  // cacheline-isolated (see static_asserts above).
  if (rank==0 && BACKEND==2) {
    if (tid < 32) {
      register_range_by_line_warp0(tile_ptr, (size_t)actual_elems * sizeof(float));
      register_range_by_line_warp0(flag_ptr, sizeof(int));
    }
    __syncwarp();
  }

  cluster.sync();
#endif

  for(int r=0; r<repeat; r++){
    // Producer: rank0 writes ACTUAL elements, then sets flag=1
    int target_val = r + 1;
    if(rank==0){
      // 1. 写入数据
      for(int i=tid; i<actual_elems; i+=blockDim.x){
        float v = (float)(i + r + cluster_id) * 0.001f;
        #if BACKEND == 1
                tile_ptr[i] = v;
        #else
                WRITE_F32(&tile_ptr[i], v);
        #endif
      }

      __syncthreads(); // 确保数据写完

      if(tid==0){
#if BACKEND == 1
        atomicExch(flag_ptr, target_val);
#else
        WRITE_S32(flag_ptr, target_val);
        __threadfence();
#endif
      }
    }

    // 确保 flag 写入对所有 rank 可见（避免死锁：多个消费者可能看不到 flag 更新）
    cluster.sync();



    // Consumers: wait flag==1, read ACTUAL elements
    if(rank!=0){
      if(tid==0){
#if BACKEND == 1
        while(atomicAdd(flag_ptr,0) < target_val) {

        }
#else
        while(READ_S32(flag_ptr) < target_val) {
            // 【关键】防止 Livelock：
            // 让出一点内存带宽，让 Producer 的写操作能挤进去
            //__threadfence();
        }
#endif
      }
      __syncthreads();

      float acc = 0.f;
      for(int i=tid; i<actual_elems; i+=blockDim.x){
#if BACKEND == 1
        acc += tile_ptr[i];
#else
        acc += READ_F32(&tile_ptr[i]);
#endif
      }
      if(tid==0) out[(size_t)cluster_id * CLUSTER_SIZE + (size_t)rank] = acc;
    }

    // Ack: last rank clears flag to 0
    cluster.sync();

  }
#endif
}

int main(int argc, char** argv){
  int cluster_size = 2;
  int repeat = 2000;
  int num_clusters = 256;  // make this big so occupancy affects total time
  int actual_elems = 512;  // fixed real moved elements per epoch

  if(argc >= 2) cluster_size = std::atoi(argv[1]);
  if(argc >= 3) repeat       = std::atoi(argv[2]);
  if(argc >= 4) num_clusters = std::atoi(argv[3]);
  if(argc >= 5) actual_elems = std::atoi(argv[4]);
  if(num_clusters < 1) num_clusters = 1;

  float* d_tiles = nullptr;
  int*   d_flags = nullptr;
  float* d_out   = nullptr;

  size_t tiles_elems = (size_t)num_clusters * (size_t)STRIDE_ELEMS;
  CUDA_CHECK(cudaMalloc(&d_tiles, tiles_elems * sizeof(float)));
  size_t flags_elems = (size_t)num_clusters * (size_t)FLAG_STRIDE_INTS;
  CUDA_CHECK(cudaMalloc(&d_flags, flags_elems * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out,   (size_t)num_clusters * (size_t)cluster_size * sizeof(float)));

  CUDA_CHECK(cudaMemset(d_tiles, 0, tiles_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_flags, 0, flags_elems * sizeof(int)));

  dim3 grid(cluster_size * num_clusters, 1, 1);
  dim3 block(256, 1, 1);

  // Warmup (comment out if you want only one kernel in sim logs)
  /*
  switch(cluster_size){
    case 2: fa_tile_many_clusters<2><<<grid,block>>>(d_tiles,d_flags,d_out, 1, actual_elems); break;
    case 4: fa_tile_many_clusters<4><<<grid,block>>>(d_tiles,d_flags,d_out, 1, actual_elems); break;
    case 8: fa_tile_many_clusters<8><<<grid,block>>>(d_tiles,d_flags,d_out, 1, actual_elems); break;
    default:
      std::fprintf(stderr, "cluster_size must be 2/4/8\n");
      return 1;
  }
    */

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));

  switch(cluster_size){
    case 2: fa_tile_many_clusters<2><<<grid,block>>>(d_tiles,d_flags,d_out, repeat, actual_elems); break;
    case 4: fa_tile_many_clusters<4><<<grid,block>>>(d_tiles,d_flags,d_out, repeat, actual_elems); break;
    case 8: fa_tile_many_clusters<8><<<grid,block>>>(d_tiles,d_flags,d_out, repeat, actual_elems); break;
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

  std::printf("BACKEND=%d cluster=%d num_clusters=%d repeat=%d RESERVE=%d ACTUAL=%d STRIDE=%d "
#if BACKEND==2
              "READ_MODE=%d WRITE_MODE=%d "
#endif
              "time_ms=%.3f\n",
              BACKEND, cluster_size, num_clusters, repeat, RESERVE_ELEMS, actual_elems, STRIDE_ELEMS
#if BACKEND==2
              , READ_MODE, WRITE_MODE
#endif
              , ms);

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_flags));
  CUDA_CHECK(cudaFree(d_tiles));
  return 0;
}