// resc_dynamic_mailbox.cu
//
// A stronger benchmark for RESC vs GLOBAL vs DSMEM.
//
// It models a cluster-local dynamic mailbox / routing-metadata pattern:
//   - rank0 in each cluster acts as a producer / scheduler
//   - every round, rank0 generates a different amount of metadata/payload for
//     each consumer block in the same cluster
//   - DSMEM must reserve worst-case shared-memory space up front
//   - GLOBAL and RESC allocate no SMEM for the mailbox payload
//   - RESC can keep the mailbox/metadata in the cluster-local coherent cache
//
// BACKEND:
//   0 = GLOBAL (.cg)
//   1 = DSMEM  (cluster.map_shared_rank to rank0's SMEM)
//   2 = RESC   (register coherent addrs + selectable ld.ca/ld.cg, st.wb/st.cg)
//
// Suggested use:
//   nvcc -arch=sm_90a -DBACKEND=1 -DRESERVE_ELEMS=8192 resc_dynamic_mailbox.cu -o bench_dsm
//   nvcc -arch=sm_90a -DBACKEND=0 -DRESERVE_ELEMS=8192 resc_dynamic_mailbox.cu -o bench_global
//   nvcc -arch=sm_90a -DBACKEND=2 -DRESERVE_ELEMS=8192 -DREAD_MODE=1 -DWRITE_MODE=1 resc_dynamic_mailbox.cu -o bench_resc
//
// Run:
//   ./bench_resc 4 2000 256 128 16 4
//   args: <cluster_size:2|4|8> <repeat> <num_clusters> <avg_elems> <burst_period> <reuse_passes>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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
// Per-cluster global/RESC payload allocation.
// Keep this fixed across sweeps to avoid changing the global footprint.
#define STRIDE_ELEMS 16384
#endif

#ifndef RESC_LINE_BYTES
#define RESC_LINE_BYTES 128
#endif

#ifndef MAX_CLUSTER_SIZE
#define MAX_CLUSTER_SIZE 8
#endif

#ifndef BURST_FACTOR
// The dynamic hot working set may occasionally expand to avg_elems * BURST_FACTOR,
// but DSMEM must still reserve RESERVE_ELEMS at compile time.
#define BURST_FACTOR 8
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

struct __align__(RESC_LINE_BYTES) ClusterMeta {
  int flags[MAX_CLUSTER_SIZE];
  int counts[MAX_CLUSTER_SIZE];
  int offsets[MAX_CLUSTER_SIZE];
  int total_elems;
  int epoch;
  int pad[6];   // 26 ints total -> 104 B, +24 B = 128 B
};
static_assert(sizeof(ClusterMeta) == RESC_LINE_BYTES,
              "ClusterMeta must occupy exactly one cache line");
static_assert((STRIDE_ELEMS * (int)sizeof(float)) % RESC_LINE_BYTES == 0,
              "STRIDE_ELEMS*sizeof(float) must be cacheline aligned");

// ---- PTX helpers ----
__device__ __forceinline__ int ld_cg_s32(const int* p){ int v;
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

__device__ __forceinline__ int ld_ca_s32(const int* p){ int v;
#if __CUDA_ARCH__ >= 900
  asm volatile("ld.global.ca.s32 %0,[%1];" : "=r"(v) : "l"(p));
#else
  v=*p;
#endif
  return v;
}
__device__ __forceinline__ float ld_ca_f32(const float* p){ float v;
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

__device__ __forceinline__ unsigned mix32(unsigned x){
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
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

#if BACKEND == 1
  #define META_READ_S32(p)   (*(p))
  #define META_WRITE_S32(p,v) do{ *(p) = (v); }while(0)
  #define PAYLOAD_READ_F32(p) (*(p))
  #define PAYLOAD_WRITE_F32(p,v) do{ *(p) = (v); }while(0)
  #define FLAG_READ_S32(p) atomicAdd((int*)(p), 0)
  #define FLAG_WRITE_S32(p,v) do{ atomicExch((int*)(p), (v)); }while(0)
#else
  #define META_READ_S32(p)   READ_S32(p)
  #define META_WRITE_S32(p,v) WRITE_S32(p,v)
  #define PAYLOAD_READ_F32(p) READ_F32(p)
  #define PAYLOAD_WRITE_F32(p,v) WRITE_F32(p,v)
  #define FLAG_READ_S32(p) READ_S32(p)
  #define FLAG_WRITE_S32(p,v) WRITE_S32(p,v)
#endif

template<int CLUSTER_SIZE>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1)
void dynamic_mailbox_many_clusters(float* __restrict__ g_payload,
                                   ClusterMeta* __restrict__ g_meta,
                                   float* __restrict__ out,
                                   int repeat,
                                   int avg_elems,
                                   int burst_period,
                                   int reuse_passes)
{
#if __CUDA_ARCH__ >= 900
  static_assert(CLUSTER_SIZE <= MAX_CLUSTER_SIZE, "cluster size exceeds MAX_CLUSTER_SIZE");

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid  = threadIdx.x;
  const int cluster_id = (int)(blockIdx.x / CLUSTER_SIZE);

  if (avg_elems < 1) avg_elems = 1;
  if (reuse_passes < 1) reuse_passes = 1;
  if (burst_period < 1) burst_period = 1;

  const int consumers = CLUSTER_SIZE - 1;
  const int hotset_elems = max(1, min(min(STRIDE_ELEMS, RESERVE_ELEMS), avg_elems * BURST_FACTOR));

  float* payload = nullptr;
  ClusterMeta* meta = nullptr;

#if BACKEND == 1
  __shared__ float smem_payload[RESERVE_ELEMS];
  __shared__ ClusterMeta smem_meta;
  payload = (float*)cluster.map_shared_rank(smem_payload, 0);
  meta    = (ClusterMeta*)cluster.map_shared_rank(&smem_meta, 0);
#else
  payload = g_payload + (size_t)cluster_id * (size_t)STRIDE_ELEMS;
  meta    = g_meta + (size_t)cluster_id;
#endif

  if (rank == 0 && tid == 0) {
    for (int i = 0; i < MAX_CLUSTER_SIZE; ++i) {
      FLAG_WRITE_S32(&meta->flags[i], 0);
      META_WRITE_S32(&meta->counts[i], 0);
      META_WRITE_S32(&meta->offsets[i], 0);
    }
    META_WRITE_S32(&meta->total_elems, 0);
    META_WRITE_S32(&meta->epoch, 0);
  }

  if (rank == 0 && BACKEND == 2) {
    if (tid < 32) {
      register_range_by_line_warp0(meta, sizeof(ClusterMeta));
      register_range_by_line_warp0(payload, (size_t)hotset_elems * sizeof(float));
    }
    __syncwarp();
  }

  cluster.sync();

  for (int r = 0; r < repeat; ++r) {
    const int target_epoch = r + 1;

    if (rank == 0) {
      if (tid == 0) {
        int cursor = 0;
        const int base_per_consumer = max(1, avg_elems / max(1, consumers));

        for (int dst = 1; dst < CLUSTER_SIZE; ++dst) {
          unsigned seed = mix32((unsigned)(cluster_id + 1) * 1315423911u ^
                                (unsigned)(r + 11) * 2654435761u ^
                                (unsigned)(dst + 7) * 2246822519u);
          const int remain_consumers = CLUSTER_SIZE - dst;
          int remaining = hotset_elems - cursor;
          int max_for_this = (remain_consumers > 0) ? (remaining / remain_consumers) : remaining;
          if (max_for_this < 0) max_for_this = 0;

          int cnt = 0;
          if ((seed & 0x7u) == 0u) {
            cnt = 0; // sparse consumer this round
          } else {
            int small = 1 + (int)(seed % (unsigned)(base_per_consumer * 2));
            int burst = min(max_for_this, max(1, base_per_consumer * BURST_FACTOR + (int)(seed % (unsigned)max(1, base_per_consumer * 2))));
            bool is_burst = ((r + cluster_id + dst) % burst_period) == 0;
            cnt = is_burst ? burst : min(max_for_this, small);
          }

          META_WRITE_S32(&meta->offsets[dst], cursor);
          META_WRITE_S32(&meta->counts[dst], cnt);
          cursor += cnt;
        }

        META_WRITE_S32(&meta->total_elems, cursor);
        META_WRITE_S32(&meta->epoch, target_epoch);
      }

      __syncthreads();

      const int total = META_READ_S32(&meta->total_elems);
      for (int i = tid; i < total; i += blockDim.x) {
        float v = (float)((i + 1) ^ (cluster_id << 4) ^ (r << 1)) * 0.001f;
        PAYLOAD_WRITE_F32(&payload[i], v);
      }

      __syncthreads();

      if (tid == 0) {
#if BACKEND != 1
        __threadfence();  // publish payload + metadata before flags become visible
#endif
        for (int dst = 1; dst < CLUSTER_SIZE; ++dst) {
          FLAG_WRITE_S32(&meta->flags[dst], target_epoch);
        }
      }
    }

    if (rank != 0) {
      if (tid == 0) {
        while (FLAG_READ_S32(&meta->flags[rank]) < target_epoch) {
        }
      }
      __syncthreads();

      float acc = 0.f;
      for (int pass = 0; pass < reuse_passes; ++pass) {
        const int count  = META_READ_S32(&meta->counts[rank]);
        const int offset = META_READ_S32(&meta->offsets[rank]);
        for (int i = tid; i < count; i += blockDim.x) {
          acc += PAYLOAD_READ_F32(&payload[offset + i]);
        }
      }

      if (tid == 0) {
        out[(size_t)cluster_id * CLUSTER_SIZE + (size_t)rank] = acc;
      }
    }

    // Keep only one cluster-wide synchronization per round: this separates rounds
    // but still lets the publish/poll/read path dominate over the synchronization cost.
    cluster.sync();
  }
#endif
}

static void print_usage(const char* prog){
  std::fprintf(stderr,
      "Usage (named args):\n"
      "  %s cluster_size=4 repeat=2000 num_clusters=256 avg_elems=128 burst_period=16 reuse_passes=4\n"
      "\n"
      "Supported parameters:\n"
      "  cluster_size : 2 | 4 | 8\n"
      "  repeat       : number of rounds per cluster\n"
      "  num_clusters : number of clusters to launch\n"
      "  avg_elems    : average payload elements generated per round\n"
      "  burst_period : every N rounds, some consumers see a bursty larger mailbox\n"
      "  reuse_passes : how many times a consumer rereads its mailbox metadata/payload\n"
      "\n"
      "Legacy positional args are still accepted in this order:\n"
      "  %s <cluster_size> <repeat> <num_clusters> <avg_elems> <burst_period> <reuse_passes>\n",
      prog, prog);
}

static bool parse_named_int_arg(const char* arg, const char* name, int* out){
  const size_t n = std::strlen(name);
  if (std::strncmp(arg, name, n) != 0 || arg[n] != '=') return false;
  *out = std::atoi(arg + n + 1);
  return true;
}

int main(int argc, char** argv){
  int cluster_size = 2;
  int repeat = 2000;
  int num_clusters = 256;
  int avg_elems = 128;
  int burst_period = 16;
  int reuse_passes = 4;

  bool saw_named_arg = false;
  bool saw_positional_arg = false;

  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
    if (std::strchr(arg, '=') != nullptr) {
      saw_named_arg = true;
      bool matched = false;
      matched = matched || parse_named_int_arg(arg, "cluster_size", &cluster_size);
      matched = matched || parse_named_int_arg(arg, "repeat", &repeat);
      matched = matched || parse_named_int_arg(arg, "num_clusters", &num_clusters);
      matched = matched || parse_named_int_arg(arg, "avg_elems", &avg_elems);
      matched = matched || parse_named_int_arg(arg, "burst_period", &burst_period);
      matched = matched || parse_named_int_arg(arg, "reuse_passes", &reuse_passes);
      if (!matched) {
        std::fprintf(stderr, "Unknown named argument: %s\n", arg);
        print_usage(argv[0]);
        return 1;
      }
    } else {
      saw_positional_arg = true;
    }
  }

  if (saw_named_arg && saw_positional_arg) {
    std::fprintf(stderr, "Please use either named arguments or legacy positional arguments, not both.\n");
    print_usage(argv[0]);
    return 1;
  }

  if (!saw_named_arg) {
    if(argc >= 2) cluster_size  = std::atoi(argv[1]);
    if(argc >= 3) repeat        = std::atoi(argv[2]);
    if(argc >= 4) num_clusters  = std::atoi(argv[3]);
    if(argc >= 5) avg_elems     = std::atoi(argv[4]);
    if(argc >= 6) burst_period  = std::atoi(argv[5]);
    if(argc >= 7) reuse_passes  = std::atoi(argv[6]);
  }

  if (cluster_size != 2 && cluster_size != 4 && cluster_size != 8) {
    std::fprintf(stderr, "cluster_size must be 2/4/8\n");
    return 1;
  }
  if (num_clusters < 1) num_clusters = 1;
  if (avg_elems < 1) avg_elems = 1;
  if (burst_period < 1) burst_period = 1;
  if (reuse_passes < 1) reuse_passes = 1;

  float* d_payload = nullptr;
  ClusterMeta* d_meta = nullptr;
  float* d_out = nullptr;

  const size_t payload_elems = (size_t)num_clusters * (size_t)STRIDE_ELEMS;
  CUDA_CHECK(cudaMalloc(&d_payload, payload_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_meta, (size_t)num_clusters * sizeof(ClusterMeta)));
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)num_clusters * (size_t)cluster_size * sizeof(float)));

  CUDA_CHECK(cudaMemset(d_payload, 0, payload_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_meta, 0, (size_t)num_clusters * sizeof(ClusterMeta)));
  CUDA_CHECK(cudaMemset(d_out, 0, (size_t)num_clusters * (size_t)cluster_size * sizeof(float)));

  dim3 grid(cluster_size * num_clusters, 1, 1);
  dim3 block(256, 1, 1);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));

  switch(cluster_size){
    case 2: dynamic_mailbox_many_clusters<2><<<grid,block>>>(d_payload, d_meta, d_out, repeat, avg_elems, burst_period, reuse_passes); break;
    case 4: dynamic_mailbox_many_clusters<4><<<grid,block>>>(d_payload, d_meta, d_out, repeat, avg_elems, burst_period, reuse_passes); break;
    case 8: dynamic_mailbox_many_clusters<8><<<grid,block>>>(d_payload, d_meta, d_out, repeat, avg_elems, burst_period, reuse_passes); break;
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

  std::printf("BACKEND=%d cluster=%d num_clusters=%d repeat=%d RESERVE=%d STRIDE=%d avg_elems=%d burst_period=%d reuse=%d hotset=%d "
#if BACKEND==2
              "READ_MODE=%d WRITE_MODE=%d "
#endif
              "time_ms=%.3f\n",
              BACKEND, cluster_size, num_clusters, repeat, RESERVE_ELEMS, STRIDE_ELEMS,
              avg_elems, burst_period, reuse_passes,
              max(1, min(min(STRIDE_ELEMS, RESERVE_ELEMS), avg_elems * BURST_FACTOR))
#if BACKEND==2
              , READ_MODE, WRITE_MODE
#endif
              , ms);

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_meta));
  CUDA_CHECK(cudaFree(d_payload));
  return 0;
}
