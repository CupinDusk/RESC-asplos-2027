// fa_tile_1toN.cu
// nvcc -O3 -arch=sm_90a fa_tile_1toN.cu -o fa -DBACKEND=2 -DRESERVE_ELEMS=4096
// ./fa 2 2000   # cluster_size=2 repeat=2000

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef BACKEND
#define BACKEND 0 // 0=GLOBAL(.cg), 1=DSMEM, 2=RESC(registered + .wb/.ca)
#endif
#ifndef RESERVE_ELEMS
#define RESERVE_ELEMS 4096
#endif

#ifndef RESC_LINE_BYTES
#define RESC_LINE_BYTES 128   // 你 sim 里如果是 64B，就改成 64
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){                       \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,              \
  cudaGetErrorString(e)); std::exit(1);} }while(0)

__device__ int g_turn;

// ---- PTX helpers (与您现有代码一致风格) ----
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

// 按 128B cacheline 注册整个通信 region（强烈建议用于数组）
//__device__ __forceinline__ void register_range_128B(void* base, int num, int bytes){
//  unsigned long long p = (unsigned long long)base;
//  unsigned long long e = p + (unsigned long long)(num*bytes);
//  for(; p < e; p += (unsigned long long)bytes) gpgpusim_register_cluster_coherent_addr((void*)p);
//}

__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
    return x & ~((uintptr_t)RESC_LINE_BYTES - 1);
  }

__device__ __forceinline__ void register_range_by_line_warp0(void* base, size_t bytes) {
    uintptr_t b = (uintptr_t)base;
    uintptr_t start = align_down_line(b);
    int nlines = (int)(( (b + bytes - start) + RESC_LINE_BYTES - 1) / RESC_LINE_BYTES);

    int lane = threadIdx.x & 31; // warp0 lanes 0..31
    for (int off = 0; off < nlines; off += 32) {
      int idx = off + lane;
      // 关键：不使用 if(idx<nlines) 来避免 lane divergence
      uintptr_t line = start + (uintptr_t)((idx < nlines) ? idx : 0) * RESC_LINE_BYTES;
      gpgpusim_register_cluster_coherent_addr((void*)line);
    }
  }

template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void fa_tile(float* __restrict__ g_buf, float* __restrict__ out, int repeat){
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  int tid  = threadIdx.x;

  // DSMEM：每个 CTA 都会分配 RESERVE_ELEMS 大小的 shared（这正是“预留痛点”的放大器
  #if BACKEND == 1
  __shared__ float smem_tile[RESERVE_ELEMS];
  __shared__ int   smem_flag;
  #endif

  float* tile_ptr = nullptr;
  int*   flag_ptr = nullptr;

#if BACKEND == 1
  tile_ptr = (float*)cluster.map_shared_rank(smem_tile, 0);
  flag_ptr = (int*)  cluster.map_shared_rank(&smem_flag, 0);
  if (tid==0 && rank==0) atomicExch(flag_ptr, 0);
  cluster.sync();
#else
  tile_ptr = g_buf;
  flag_ptr = &g_turn;

  /*if (rank==0 && BACKEND==2) {
    if (tid < 32) {
      register_range_by_line_warp0(tile_ptr, RESERVE_ELEMS * sizeof(float));
      register_range_by_line_warp0(flag_ptr, sizeof(int));
    }
    __syncwarp();
  }
  cluster.sync();
  */

#endif

if (rank==0) {
    if (tid < 32) {
      register_range_by_line_warp0(tile_ptr, RESERVE_ELEMS * sizeof(float));
      register_range_by_line_warp0(flag_ptr, sizeof(int));
    }
    __syncwarp();
  }

  cluster.sync();

  for(int r=0;r<repeat;r++){
    // Producer: rank0 写 tile，发布 flag=1
    if(rank==0){
        if (tid == 0) {
#if BACKEND == 1
      while(atomicAdd(flag_ptr,0)!=0){}
#else
      while((BACKEND==2? ld_ca_s32(flag_ptr): ld_cg_s32(flag_ptr))!=0){}
#endif
        }
        __syncthreads();

      for(int i=tid;i<RESERVE_ELEMS;i+=blockDim.x){
        float v = (float)(i + r) * 0.001f;
#if BACKEND == 2
        //st_wb_f32(&tile_ptr[i], v);
        st_cg_f32(&tile_ptr[i], v);
#elif BACKEND == 0
        st_cg_f32(&tile_ptr[i], v);
#else
        tile_ptr[i] = v;
#endif
      }
      __syncthreads();
#if BACKEND == 1
      atomicExch(flag_ptr, 1);
#elif BACKEND == 2
      //st_wb_s32(flag_ptr, 1);
      st_cg_s32(flag_ptr, 1);
#else
      st_cg_s32(flag_ptr, 1);
#endif
    }

    // Consumers: rank>0 读 tile
    if(rank!=0){
        if(tid == 0) {
#if BACKEND == 1
      while(atomicAdd(flag_ptr,0)!=1){}
#else
      while((BACKEND==2? ld_ca_s32(flag_ptr): ld_cg_s32(flag_ptr))!=1){}
#endif
        }
        __syncthreads();

      float acc=0.f;
      for(int i=tid;i<RESERVE_ELEMS;i+=blockDim.x){
#if BACKEND == 2
        acc += ld_ca_f32(&tile_ptr[i]);
#elif BACKEND == 0
        acc += ld_cg_f32(&tile_ptr[i]);
#else
        acc += tile_ptr[i];
#endif
      }
      if(tid==0) out[rank] = acc;
    }

    // 最后一个 rank 清 flag=0（ack），避免多消费者原子
    cluster.sync();
    if(rank==CLUSTER_SIZE-1 && tid==0){
#if BACKEND == 1
      atomicExch(flag_ptr, 0);
#elif BACKEND == 2
      //st_wb_s32(flag_ptr, 0);
      st_cg_s32(flag_ptr, 0);
#else
      st_cg_s32(flag_ptr, 0);
#endif
    }
    cluster.sync();
  }
#endif
}

int main(int argc,char** argv){
  int cluster_size=2, repeat=2000;
  if(argc>=2) cluster_size=std::atoi(argv[1]);
  if(argc>=3) repeat=std::atoi(argv[2]);

  float *d_buf=nullptr,*d_out=nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, RESERVE_ELEMS*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, cluster_size*sizeof(float)));

  int zero=0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_turn,&zero,sizeof(int)));

  dim3 grid(cluster_size,1,1);
  dim3 block(256,1,1);

  switch(cluster_size){
    case 2: fa_tile<2><<<grid,block>>>(d_buf,d_out,repeat); break;
    case 4: fa_tile<4><<<grid,block>>>(d_buf,d_out,repeat); break;
    case 8: fa_tile<8><<<grid,block>>>(d_buf,d_out,repeat); break;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_buf));
  return 0;
}
