// j2_headtohead.cu -- JOB 2: whole-kernel head-to-head, RESC vs the legal
// ld.ca + cluster-barrier-acquire program, with the whole-L1 invalidate charged.
//
// Structure per epoch (identical in every arm):
//   rank0 produces a per-cluster comm payload (hotset) and publishes it
//   consumers (rank != 0) read the payload REUSE times      <- comm axis
//   ALL ranks run real FMA compute over a PRIVATE UNMARKED GLOBAL tile,
//     read with ld.global.ca, TILE_PASSES times              <- unmarked-L1 axis
//   [arm-dependent acquire fence]
//   cluster.sync()
//
// The tile is GLOBAL (not SMEM) and read with .ca on purpose: a whole-L1
// invalidate destroys L1 reuse, and SMEM is never touched by CCTL.IVALL, so an
// SMEM tile cannot express the selectivity contrast at all.
//
// ARM:
//  0 cg_relaxed   comm ld.cg, no acquire fence   (legal; protects the tile)
//  1 dsm          comm in rank0 SMEM via map_shared_rank, no fence
//  2 resc         comm ld.ca + RESC marked lines, no fence  (modeled RESC; UPPER BOUND)
//  3 ca_acq       comm ld.ca + __threadfence() before cluster.sync()  <-- THE NEW ARM
//  4 cg_acq       comm ld.cg + __threadfence()  (the manuscript's current baseline)
//  5 hybrid       comm staged global->own SMEM once, reuse from SMEM, no fence
//  6 resc_faith   like 2 but the FIRST comm pass of each epoch uses ld.cg,
//                 modeling marked-line self-invalidation at epoch close
//                 (one compulsory miss, then L1-resident) = FAITHFUL RESC
//
// In this simulator __threadfence() (membar.gl) is wired to a whole-L1
// invalidate when -gpgpu_flush_l1_cache 1 (shader.cc:4069-4080), which is
// exactly CCTL.IVALL semantics.  Producer publishes use st.global.cg rather
// than a fence so that the ONLY invalidate in the kernel is the arm-controlled
// acquire.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
namespace cg = cooperative_groups;

#ifndef ARM
#define ARM 0
#endif
#ifndef CS
#define CS 8
#endif
#ifndef TILE_KB
#define TILE_KB 64
#endif
#ifndef REUSE
#define REUSE 4
#endif
#ifndef TILE_PASSES
#define TILE_PASSES 4
#endif
#ifndef EPOCHS
#define EPOCHS 64
#endif
#ifndef NCLUSTERS
#define NCLUSTERS 8
#endif
#ifndef HOTSET_ELEMS
#define HOTSET_ELEMS 1024          /* 4 KB comm payload */
#endif
#ifndef STRIDE_ELEMS
#define STRIDE_ELEMS 4096
#endif
#ifndef RESC_LINE_BYTES
#define RESC_LINE_BYTES 128
#endif
#ifndef MAX_CLUSTER_SIZE
#define MAX_CLUSTER_SIZE 8
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(1);} }while(0)

struct __align__(RESC_LINE_BYTES) ClusterMeta {
  int flags[MAX_CLUSTER_SIZE];
  int counts[MAX_CLUSTER_SIZE];
  int offsets[MAX_CLUSTER_SIZE];
  int total_elems; int epoch; int pad[6];
};
static_assert(sizeof(ClusterMeta) == RESC_LINE_BYTES, "meta must be one line");

__device__ __forceinline__ int   ld_cg_s32(const int* p){ int v;
  asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ void  st_cg_s32(int* p,int v){
  asm volatile("st.global.cg.s32 [%0],%1;" :: "l"(p),"r"(v) : "memory"); }
__device__ __forceinline__ float ld_cg_f32(const float* p){ float v;
  asm volatile("ld.global.cg.f32 %0,[%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ void  st_cg_f32(float* p,float v){
  asm volatile("st.global.cg.f32 [%0],%1;" :: "l"(p),"f"(v) : "memory"); }
__device__ __forceinline__ int   ld_ca_s32(const int* p){ int v;
  asm volatile("ld.global.ca.s32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ float ld_ca_f32(const float* p){ float v;
  asm volatile("ld.global.ca.f32 %0,[%1];" : "=f"(v) : "l"(p)); return v; }

// RESC marking hook (intercepted by the simulator)
__device__ volatile int g_dummy_registration_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_dummy_registration_flag = (int)((unsigned long long)addr & 0x1); (void)addr; }
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){
  return x & ~((uintptr_t)RESC_LINE_BYTES - 1); }
__device__ __forceinline__ void register_range_by_line_warp0(void* base, size_t bytes){
  uintptr_t b=(uintptr_t)base, start=align_down_line(b);
  int nlines=(int)(((b+bytes-start)+RESC_LINE_BYTES-1)/RESC_LINE_BYTES);
  int lane=threadIdx.x & 31;
  for(int off=0; off<nlines; off+=32){
    int idx=off+lane;
    uintptr_t line=start+(uintptr_t)((idx<nlines)?idx:0)*RESC_LINE_BYTES;
    gpgpusim_register_cluster_coherent_addr((void*)line);
  }
}

// ---- arm properties ----
#define COMM_IS_DSM     (ARM==1)
#define COMM_IS_HYBRID  (ARM==5)
#define COMM_READ_CA    (ARM==2 || ARM==3 || ARM==6)
#define DO_ACQ_FENCE    (ARM==3 || ARM==4)
#define DO_RESC_MARK    (ARM==2 || ARM==6)
#define RESC_FAITHFUL   (ARM==6)

__global__ __cluster_dims__(CS,1,1)
void headtohead(float* __restrict__ g_payload, ClusterMeta* __restrict__ g_meta,
                float* __restrict__ g_tile, float* __restrict__ out)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid  = threadIdx.x;
  const int blk  = blockIdx.x;
  const int cluster_id = blk / CS;
  const int TILE_ELEMS = (TILE_KB*1024)/(int)sizeof(float);
  const float* mytile = g_tile + (size_t)blk * (size_t)TILE_ELEMS;

  float* payload; ClusterMeta* meta;
#if COMM_IS_DSM
  __shared__ float smem_payload[HOTSET_ELEMS];
  __shared__ ClusterMeta smem_meta;
  payload = (float*)cluster.map_shared_rank(smem_payload, 0);
  meta    = (ClusterMeta*)cluster.map_shared_rank(&smem_meta, 0);
#else
  payload = g_payload + (size_t)cluster_id * (size_t)STRIDE_ELEMS;
  meta    = g_meta + cluster_id;
#endif
#if COMM_IS_HYBRID
  __shared__ float stage[HOTSET_ELEMS];
#endif

  if (rank==0 && tid==0) {
    for (int i=0;i<MAX_CLUSTER_SIZE;++i){
#if COMM_IS_DSM
      atomicExch((int*)&meta->flags[i],0); meta->counts[i]=0; meta->offsets[i]=0;
#else
      st_cg_s32(&meta->flags[i],0); st_cg_s32(&meta->counts[i],0); st_cg_s32(&meta->offsets[i],0);
#endif
    }
  }
#if DO_RESC_MARK
  if (rank==0 && tid<32) {
    register_range_by_line_warp0(meta, sizeof(ClusterMeta));
    register_range_by_line_warp0(payload, (size_t)HOTSET_ELEMS*sizeof(float));
  }
  __syncwarp();
#endif
  cluster.sync();

  float acc = 0.f, cacc = 0.f;
  const int consumers = CS-1;
  const int per_consumer = HOTSET_ELEMS / (consumers>0?consumers:1);

  for (int e=0; e<EPOCHS; ++e) {
    const int target = e+1;

    // ---- produce + publish ----
    if (rank==0) {
      if (tid==0) {
        int cursor=0;
        for (int dst=1; dst<CS; ++dst) {
#if COMM_IS_DSM
          meta->offsets[dst]=cursor; meta->counts[dst]=per_consumer;
#else
          st_cg_s32(&meta->offsets[dst],cursor); st_cg_s32(&meta->counts[dst],per_consumer);
#endif
          cursor += per_consumer;
        }
#if COMM_IS_DSM
        meta->total_elems=cursor; meta->epoch=target;
#else
        st_cg_s32(&meta->total_elems,cursor); st_cg_s32(&meta->epoch,target);
#endif
      }
      __syncthreads();
      for (int i=tid; i<HOTSET_ELEMS; i+=blockDim.x) {
        float v = (float)((i+1) ^ (cluster_id<<4) ^ (e<<1)) * 0.001f;
#if COMM_IS_DSM
        payload[i]=v;
#else
        st_cg_f32(&payload[i],v);
#endif
      }
      __syncthreads();
      if (tid==0) for (int dst=1; dst<CS; ++dst) {
#if COMM_IS_DSM
        atomicExch((int*)&meta->flags[dst], target);
#else
        st_cg_s32(&meta->flags[dst],target);
#endif
      }
    }

    // ---- consume ----
    if (rank!=0) {
      if (tid==0) {
#if COMM_IS_DSM
        while (atomicAdd((int*)&meta->flags[rank], 0) < target) {}
#else
        while (ld_cg_s32(&meta->flags[rank]) < target) {}
#endif
      }
      __syncthreads();
#if COMM_IS_DSM
      const int count=meta->counts[rank], offset=meta->offsets[rank];
#else
      const int count=ld_cg_s32(&meta->counts[rank]), offset=ld_cg_s32(&meta->offsets[rank]);
#endif
#if COMM_IS_HYBRID
      for (int i=tid;i<count;i+=blockDim.x) stage[i]=ld_cg_f32(&payload[offset+i]);
      __syncthreads();
      for (int p=0;p<REUSE;++p) for (int i=tid;i<count;i+=blockDim.x) acc += stage[i];
#else
      for (int p=0;p<REUSE;++p) {
        for (int i=tid;i<count;i+=blockDim.x) {
#if COMM_IS_DSM
          acc += payload[offset+i];
#elif RESC_FAITHFUL
          // marked-line self-invalidation at epoch close: first pass refetches
          acc += (p==0) ? ld_cg_f32(&payload[offset+i]) : ld_ca_f32(&payload[offset+i]);
#elif COMM_READ_CA
          acc += ld_ca_f32(&payload[offset+i]);
#else
          acc += ld_cg_f32(&payload[offset+i]);
#endif
        }
      }
#endif
    }

    // ---- real compute over the PRIVATE UNMARKED GLOBAL tile (all ranks) ----
    {
      float c = (float)(rank+e)*1e-3f + 1.0f;
      for (int p=0;p<TILE_PASSES;++p) {
        for (int i=tid;i<TILE_ELEMS;i+=blockDim.x) {
          float t = ld_ca_f32(&mytile[i]);
          c = fmaf(t, c, 0.5f);
        }
      }
      cacc += c;
    }

    // ---- arm-controlled acquire ----
#if DO_ACQ_FENCE
    __threadfence();     // membar.gl -> whole-L1 invalidate (= CCTL.IVALL)
#endif
    cluster.sync();
  }

  if (tid==0) out[blk] = acc + cacc;
#endif
}

int main(){
  const int TILE_ELEMS=(TILE_KB*1024)/(int)sizeof(float);
  const int nblocks=CS*NCLUSTERS;
  float *d_payload=nullptr,*d_tile=nullptr,*d_out=nullptr; ClusterMeta* d_meta=nullptr;
  size_t pe=(size_t)NCLUSTERS*STRIDE_ELEMS;
  size_t te=(size_t)nblocks*TILE_ELEMS;
  CUDA_CHECK(cudaMalloc(&d_payload,pe*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_meta,(size_t)NCLUSTERS*sizeof(ClusterMeta)));
  CUDA_CHECK(cudaMalloc(&d_tile,te*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out,(size_t)nblocks*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_payload,0,pe*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_meta,0,(size_t)NCLUSTERS*sizeof(ClusterMeta)));
  CUDA_CHECK(cudaMemset(d_tile,0,te*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out,0,(size_t)nblocks*sizeof(float)));

  cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a));
  headtohead<<<nblocks,256>>>(d_payload,d_meta,d_tile,d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));

  float* h=(float*)malloc((size_t)nblocks*sizeof(float));
  CUDA_CHECK(cudaMemcpy(h,d_out,(size_t)nblocks*sizeof(float),cudaMemcpyDeviceToHost));
  double s=0; for(int i=0;i<nblocks;++i) s+=(double)h[i];
  std::printf("J2 ARM=%d CS=%d TILE_KB=%d REUSE=%d TILE_PASSES=%d EPOCHS=%d NCLUSTERS=%d "
              "HOTSET=%d nblocks=%d CHECKSUM=%.6f time_ms=%.3f\n",
              ARM,CS,TILE_KB,REUSE,TILE_PASSES,EPOCHS,NCLUSTERS,HOTSET_ELEMS,nblocks,s,ms);
  free(h);
  return 0;
}
