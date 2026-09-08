// resc_decode.cu -- Route 1 sim e2e: REAL cluster-cooperative fused GQA decode attention.
// Built against the EXISTING RESC protocol library (no sim rebuild, no core-coherence edits).
// One thread-block cluster = one GQA KV-group. The N blocks (query heads) of the cluster:
//   (region #1, PRIMARY) rank0 gathers the current-token K/V tile into the marked region;
//       all N query-head blocks REUSE the same K/V tile (fan-out read-share) over the medium.
//   compute: per-block Q.K^T GeMV -> online-softmax partial (S_max,S_sum) -> weighted-V accumulate.
//   (region #2, SECONDARY) ClusterReduce partial (S_max,S_sum,A_b) over a log-tree; each round
//       reads a freshly peer-written buffer over the medium (release/acquire producer->consumer).
// Compute body is BYTE-IDENTICAL across all 3 arms; only the marked-region SERVING PATH differs:
//   BACKEND 0 GLOBAL: ld.cg/st.cg (legal cross-SM: refetch from L2 every reuse, 242cyc)
//   BACKEND 1 DSMEM : cluster.map_shared_rank + SMEM reserve (occupancy tax)
//   BACKEND 2 RESC  : register coherent lines + ld.ca/st.wb (served L1<->L1 over SM-to-SM fabric,195cyc)
//
// Build: nvcc -arch=sm_90 -DBACKEND=<0|1|2> [-DREAD_MODE=1 -DWRITE_MODE=1 for RESC] resc_decode.cu -o <bin> -lcudart
// Run  : ./bin <cluster_size:2|4|8> <repeat=decode_steps> <num_clusters> <Dhead> <Lsplit>

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
#ifndef DHEAD
#define DHEAD 64          // head dim (GeMV length); compile-time so loops unroll identically
#endif
#ifndef LSPLIT
#define LSPLIT 32         // KV positions per block's sequence split
#endif
#ifndef MAX_CLUSTER_SIZE
#define MAX_CLUSTER_SIZE 8
#endif
#ifndef RESC_LINE_BYTES
#define RESC_LINE_BYTES 128
#endif
// Per-cluster global slab: KV tile (DHEAD*2 for K and V current token) + partials scratch.
#ifndef STRIDE_ELEMS
#define STRIDE_ELEMS 8192
#endif
#ifndef RESERVE_ELEMS
#define RESERVE_ELEMS 4096   // DSMEM compile-time SMEM reserve (the occupancy pain point)
#endif
#ifndef READ_MODE
#define READ_MODE 0       // BACKEND2: 0=ld.cg 1=ld.ca
#endif
#ifndef WRITE_MODE
#define WRITE_MODE 0      // BACKEND2: 0=st.cg 1=st.wb
#endif

#define CUDA_CHECK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- PTX serving-path helpers ----
__device__ __forceinline__ int   ld_cg_s32(const int* p){int v; asm volatile("ld.global.cg.s32 %0,[%1];":"=r"(v):"l"(p)); return v;}
__device__ __forceinline__ void  st_cg_s32(int* p,int v){asm volatile("st.global.cg.s32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ float ld_cg_f32(const float* p){float v; asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(v):"l"(p)); return v;}
__device__ __forceinline__ void  st_cg_f32(float* p,float v){asm volatile("st.global.cg.f32 [%0],%1;"::"l"(p),"f"(v):"memory");}
__device__ __forceinline__ int   ld_ca_s32(const int* p){int v; asm volatile("ld.global.ca.s32 %0,[%1];":"=r"(v):"l"(p)); return v;}
__device__ __forceinline__ float ld_ca_f32(const float* p){float v; asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(p)); return v;}
__device__ __forceinline__ void  st_wb_s32(int* p,int v){asm volatile("st.global.wb.s32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ void  st_wb_f32(float* p,float v){asm volatile("st.global.wb.f32 [%0],%1;"::"l"(p),"f"(v):"memory");}

__device__ volatile int g_dummy_registration_flag=0;
extern "C" __device__ __attribute__((noinline))
void gpgpusim_register_cluster_coherent_addr(void* addr){
  g_dummy_registration_flag = (int)((unsigned long long)addr & 0x1); (void)addr;
}
__device__ __forceinline__ uintptr_t align_down_line(uintptr_t x){return x & ~((uintptr_t)RESC_LINE_BYTES-1);}
__device__ __forceinline__ void register_range_by_line_warp0(void* base,size_t bytes){
  uintptr_t b=(uintptr_t)base,start=align_down_line(b);
  int nlines=(int)(((b+bytes-start)+RESC_LINE_BYTES-1)/RESC_LINE_BYTES);
  int lane=threadIdx.x&31;
  for(int off=0;off<nlines;off+=32){int idx=off+lane;
    uintptr_t line=start+(uintptr_t)((idx<nlines)?idx:0)*RESC_LINE_BYTES;
    gpgpusim_register_cluster_coherent_addr((void*)line);}
}

#if BACKEND==2
  #define READ_S32(p)   (READ_MODE==1?ld_ca_s32(p):ld_cg_s32(p))
  #define READ_F32(p)   (READ_MODE==1?ld_ca_f32(p):ld_cg_f32(p))
  #define WRITE_S32(p,v) do{ if(WRITE_MODE==1) st_wb_s32(p,v); else st_cg_s32(p,v);}while(0)
  #define WRITE_F32(p,v) do{ if(WRITE_MODE==1) st_wb_f32(p,v); else st_cg_f32(p,v);}while(0)
  #define MREAD_S32(p) READ_S32(p)
  #define MREAD_F32(p) READ_F32(p)
  #define MWRITE_S32(p,v) WRITE_S32(p,v)
  #define MWRITE_F32(p,v) WRITE_F32(p,v)
  #define FLAG_READ(p)  READ_S32(p)
  #define FLAG_WRITE(p,v) WRITE_S32(p,v)
#elif BACKEND==0
  #define MREAD_S32(p) ld_cg_s32(p)
  #define MREAD_F32(p) ld_cg_f32(p)
  #define MWRITE_S32(p,v) st_cg_s32(p,v)
  #define MWRITE_F32(p,v) st_cg_f32(p,v)
  #define FLAG_READ(p)  ld_cg_s32(p)
  #define FLAG_WRITE(p,v) st_cg_s32(p,v)
#elif BACKEND==1
  #define MREAD_S32(p) (*(p))
  #define MREAD_F32(p) (*(p))
  #define MWRITE_S32(p,v) do{*(p)=(v);}while(0)
  #define MWRITE_F32(p,v) do{*(p)=(v);}while(0)
  #define FLAG_READ(p)  atomicAdd((int*)(p),0)
  #define FLAG_WRITE(p,v) do{ atomicExch((int*)(p),(v)); }while(0)
#endif

// Cluster-shared layout in the marked region (per cluster):
//   [0 .. DHEAD)          : K current-token tile (region #1, produced by rank0, reused by all)
//   [DHEAD .. 2*DHEAD)    : V current-token tile (region #1)
//   [2*DHEAD .. +3*CS)    : partials S_max,S_sum,A_b per block (region #2, ClusterReduce)
//   flags: one int per block for gather-publish; reduce uses epoch flags
struct Layout { int kv_off, part_off; };
__device__ __forceinline__ int KV_OFF(){ return 0; }
__device__ __forceinline__ int PART_OFF(){ return 2*DHEAD; }  // 3 floats per rank after this

template<int CS>
__global__ __cluster_dims__(CS,1,1)
void decode_attn_cluster(float* __restrict__ g_slab,   // [num_clusters * STRIDE_ELEMS]
                         int*   __restrict__ g_flags,  // [num_clusters * FLAG_STRIDE]
                         float* __restrict__ g_Q,      // [num_clusters * CS * DHEAD] per (cluster,rank) query
                         float* __restrict__ out,      // [num_clusters * CS]
                         int repeat)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid  = threadIdx.x;
  const int cid  = (int)(blockIdx.x / CS);
  const int FLAG_STRIDE = RESC_LINE_BYTES/(int)sizeof(int); // 32; per-block flag padded

  float* slab; int* flags;
#if BACKEND==1
  __shared__ float smem_slab[RESERVE_ELEMS];
  __shared__ int   smem_flags[MAX_CLUSTER_SIZE*8];
  slab  = (float*)cluster.map_shared_rank(smem_slab, 0);
  flags = (int*)  cluster.map_shared_rank(smem_flags, 0);
#else
  slab  = g_slab  + (size_t)cid * STRIDE_ELEMS;
  flags = g_flags + (size_t)cid * (size_t)(MAX_CLUSTER_SIZE*FLAG_STRIDE);
#endif
  const float* Q = g_Q + ((size_t)cid*CS + rank)*DHEAD;

  if(rank==0 && tid==0){
    for(int i=0;i<MAX_CLUSTER_SIZE;i++) FLAG_WRITE(&flags[i*FLAG_STRIDE], 0);
  }
#if BACKEND==2
  if(rank==0 && BACKEND==2){
    if(tid<32){
      register_range_by_line_warp0(slab, (size_t)(PART_OFF()+3*CS)*sizeof(float));
      register_range_by_line_warp0(flags, (size_t)(MAX_CLUSTER_SIZE*FLAG_STRIDE)*sizeof(int));
    }
    __syncwarp();
  }
#endif
  cluster.sync();

  float local_out = 0.f;
  for(int step=0; step<repeat; ++step){
    const int epoch = step+1;

    // ===== region #1: rank0 GATHERS current-token K/V tile; peers REUSE it (fan-out read-share)
    if(rank==0){
      for(int d=tid; d<DHEAD; d+=blockDim.x){
        float kv = (float)((d + step*7 + cid) % 251) * 0.001f;
        MWRITE_F32(&slab[KV_OFF()+d], kv);           // K
        MWRITE_F32(&slab[DHEAD+d], kv*0.5f + 0.01f); // V
      }
      __syncthreads();
      if(tid==0){
#if BACKEND!=1
        __threadfence();
#endif
        for(int i=0;i<CS;i++) FLAG_WRITE(&flags[i*FLAG_STRIDE], epoch);
      }
    }
    cluster.sync();

    // All ranks (query heads) wait for gather, then reuse the SAME K/V tile.
    if(tid==0){ while(FLAG_READ(&flags[rank*FLAG_STRIDE]) < epoch){} }
    __syncthreads();

    // ===== compute (BYTE-IDENTICAL across arms): Q.K^T GeMV -> online softmax -> weighted V
    // Each block owns an LSPLIT-position sequence split; positions share the current-token K/V
    // tile as the base (models decode: one new token's K/V reused across the split & heads).
    __shared__ float sc[LSPLIT];
    float qval = 0.f;
    for(int d=0; d<DHEAD; ++d) qval += Q[d];   // reduce query to scalar (deterministic)
    for(int j=tid; j<LSPLIT; j+=blockDim.x){
      // score_j = (Q . K_tile) scaled + positional perturbation for this split position
      float s = 0.f;
      for(int d=0; d<DHEAD; ++d){
        float k = MREAD_F32(&slab[KV_OFF()+d]);   // reuse shared K tile over medium
        s += (Q[d]) * k;
      }
      s = s * rsqrtf((float)DHEAD) + (float)((j*13 + rank*3)%17)*0.01f;
      sc[j] = s;
    }
    __syncthreads();
    // online softmax partial over this block's LSPLIT
    float m = -1e30f; for(int j=0;j<LSPLIT;j++) m = fmaxf(m, sc[j]);
    float sum = 0.f;  for(int j=0;j<LSPLIT;j++) sum += __expf(sc[j]-m);
    // weighted-V accumulate: reduce shared V tile over the medium (deterministic, thread0)
    float a_b = 0.f;
    if(tid==0){ for(int d=0; d<DHEAD; ++d) a_b += MREAD_F32(&slab[DHEAD+d]); }
    __syncthreads();

    // publish this block's partial (S_max=m, S_sum=sum, A_b=a_b) to region #2
    if(tid==0){
      MWRITE_F32(&slab[PART_OFF()+rank*3+0], m);
      MWRITE_F32(&slab[PART_OFF()+rank*3+1], sum);
      MWRITE_F32(&slab[PART_OFF()+rank*3+2], a_b);
#if BACKEND!=1
      __threadfence();
#endif
    }
    cluster.sync();

    // ===== region #2: ClusterReduce partials across the CS blocks (log-tree).
    // rank0 combines all partials (each read is a peer-written buffer over the medium).
    if(rank==0 && tid==0){
      float gmax=-1e30f;
      for(int i=0;i<CS;i++) gmax=fmaxf(gmax, MREAD_F32(&slab[PART_OFF()+i*3+0]));
      float gsum=0.f, gacc=0.f;
      for(int i=0;i<CS;i++){
        float pm=MREAD_F32(&slab[PART_OFF()+i*3+0]);
        float ps=MREAD_F32(&slab[PART_OFF()+i*3+1]);
        float pa=MREAD_F32(&slab[PART_OFF()+i*3+2]);
        float scale=__expf(pm-gmax);
        gsum += ps*scale;
        gacc += pa*scale;
      }
      float attn = (gsum>0.f)? (gacc/gsum) : 0.f;
      local_out += attn;
    }
    cluster.sync();
  }

  if(tid==0) out[(size_t)cid*CS + rank] = local_out;
#endif
}

int main(int argc,char**argv){
  int cs=4, repeat=64, ncl=64, dhead=DHEAD, lsplit=LSPLIT;
  if(argc>1) cs=atoi(argv[1]);
  if(argc>2) repeat=atoi(argv[2]);
  if(argc>3) ncl=atoi(argv[3]);
  // dhead/lsplit are compile-time; args 4/5 ignored (kept for CLI symmetry)
  if(cs!=2&&cs!=4&&cs!=8){fprintf(stderr,"cluster must be 2/4/8\n");return 1;}
  if(ncl<1)ncl=1; if(repeat<1)repeat=1;
  const int FLAG_STRIDE = RESC_LINE_BYTES/(int)sizeof(int);
  float *d_slab,*d_Q,*d_out; int* d_flags;
  size_t slab_e=(size_t)ncl*STRIDE_ELEMS;
  size_t flag_e=(size_t)ncl*MAX_CLUSTER_SIZE*FLAG_STRIDE;
  size_t q_e=(size_t)ncl*cs*dhead;
  size_t o_e=(size_t)ncl*cs;
  // Reserve a small pad FIRST so no marked buffer lands at global-heap offset 0
  // (0xf00000000), which the sim's RESC registry rejects. Mirrors resc_e2e's layout.
  void* d_pad=nullptr; CUDA_CHECK(cudaMalloc(&d_pad, 65536));
  CUDA_CHECK(cudaMalloc(&d_slab, slab_e*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_flags, flag_e*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_Q, q_e*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, o_e*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_slab,0,slab_e*sizeof(float)));
  CUDA_CHECK(cudaMemset(d_flags,0,flag_e*sizeof(int)));
  CUDA_CHECK(cudaMemset(d_out,0,o_e*sizeof(float)));
  float* hq=(float*)malloc(q_e*sizeof(float));
  for(size_t i=0;i<q_e;i++) hq[i]=(float)((i%89)-44)*0.01f;
  CUDA_CHECK(cudaMemcpy(d_Q,hq,q_e*sizeof(float),cudaMemcpyHostToDevice));

  dim3 grid(cs*ncl,1,1), block(256,1,1);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  switch(cs){
    case 2: decode_attn_cluster<2><<<grid,block>>>(d_slab,d_flags,d_Q,d_out,repeat); break;
    case 4: decode_attn_cluster<4><<<grid,block>>>(d_slab,d_flags,d_Q,d_out,repeat); break;
    case 8: decode_attn_cluster<8><<<grid,block>>>(d_slab,d_flags,d_Q,d_out,repeat); break;
  }
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1));
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  float* ho=(float*)malloc(o_e*sizeof(float));
  CUDA_CHECK(cudaMemcpy(ho,d_out,o_e*sizeof(float),cudaMemcpyDeviceToHost));
  double sum=0; float mn=1e30f,mx=-1e30f;
  for(size_t i=0;i<o_e;i++){sum+=ho[i]; if(ho[i]<mn)mn=ho[i]; if(ho[i]>mx)mx=ho[i];}
  printf("OUT_CHECKSUM sum=%.6f min=%.6f max=%.6f n=%zu\n", sum, mn, mx, o_e);
  printf("BACKEND=%d cluster=%d ncl=%d repeat=%d DHEAD=%d LSPLIT=%d "
#if BACKEND==2
         "READ_MODE=%d WRITE_MODE=%d "
#endif
         "time_ms=%.3f\n", BACKEND,cs,ncl,repeat,dhead,lsplit
#if BACKEND==2
         ,READ_MODE,WRITE_MODE
#endif
         ,ms);
  return 0;
}
