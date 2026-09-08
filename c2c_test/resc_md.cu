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

#ifndef TILE_KB
#define TILE_KB 0   /* dummy compute-tile dynamic SMEM (KB) reserved in ALL arms */
#endif

#ifndef COMPUTE_ELEMS
#define COMPUTE_ELEMS 0   /* 0 = full tile; else cap compute loop to this many elems (occupancy still set by TILE_KB) */
#endif
#ifndef COMPUTE_ITERS
#define COMPUTE_ITERS 0   /* real FP-compute passes over the SMEM tile per round.
                             0 = legacy O(1) touch (comm-only). >0 = whole-kernel
                             compute+comm: matmul-shaped FMA modeling the flash-attn
                             inner loop (S=Q.K^T, O+=P.V) so occupancy multiplies real work. */
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
#elif BACKEND == 4
  // Non-coherent L1-residency CONTROL: read communicated region with ld.ca
  // (L1-CACHING), no RESC protocol. Isolates bare-L1-residency benefit from the
  // coherence machinery. Safe here only because single-producer writes once
  // before any consumer reads within an epoch (would be a staleness hazard in
  // general -- that hazard is exactly what RESC's coherence makes safe).
  #define READ_S32(p)  ld_ca_s32(p)
  #define READ_F32(p)  ld_ca_f32(p)
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

// ============================================================================
// resc_md.cu : fused MD / short-range N-body, cluster-cooperative neighbor-cell
// position sharing (W3 real-kernel end-to-end for RESC).
//   Fused velocity-Verlet inner step: DRIFT (produce new positions) ->
//   cluster.sync -> FORCE (consume neighbor-cell positions).
//   cluster = CLUSTER_SIZE blocks over a compact cell brick; block = 1 cell.
//   Single-producer-per-line: each cell's positions have one producer (owner);
//   home block reads neighbor positions L1-to-L1, accumulates force on OWN
//   particles only (no reciprocal write-back -> preserves single-producer).
//   Marked communication line = per-cell new-position buffer (RESC region).
// BACKEND: 0=global(ld.cg, correct-legal baseline) 1=DSM(map_shared_rank) 2=RESC
// ============================================================================

#ifndef MAXP_PER_CELL
#define MAXP_PER_CELL 64   // worst-case particles/cell (DSM/staging must reserve this)
#endif
#ifndef CLUSTER_CELLS
#define CLUSTER_CELLS CLUSTER_SIZE  // one block per cell
#endif

// positions laid out per cell: cell c owns POS[c*MAXP_PER_CELL .. +count[c]]
// each particle = 4 floats (x,y,z,pad) so a particle is 16B (cache-line friendly)
struct __align__(16) P4 { float x,y,z,w; };

#if BACKEND == 1
// DSM: peer positions live in mapped SMEM -> plain access
__device__ __forceinline__ P4 read_pos(const P4* base, int idx){ return base[idx]; }
__device__ __forceinline__ void write_pos(P4* base, int idx, P4 v){ base[idx]=v; }
#else
__device__ __forceinline__ P4 read_pos(const P4* base, int idx){
  const float* p = (const float*)&base[idx];
  P4 r; r.x=READ_F32(p+0); r.y=READ_F32(p+1); r.z=READ_F32(p+2); r.w=0.f; return r;
}
__device__ __forceinline__ void write_pos(P4* base, int idx, P4 v){
  float* p = (float*)&base[idx];
  WRITE_F32(p+0,v.x); WRITE_F32(p+1,v.y); WRITE_F32(p+2,v.z);
}
#endif

template<int CS>
__global__ __cluster_dims__(CS,1,1)
void md_step(P4* __restrict__ g_pos,      // [num_cells * MAXP_PER_CELL]
             P4* __restrict__ g_vel,
             int* __restrict__ g_count,   // particles per cell
             const int* __restrict__ g_neigh, // [num_cells * 27] in-cluster neighbor cell ids (-1 = out/none)
             float* __restrict__ g_out,    // force checksum sink
             int num_clusters, int repeat, float dt, float rc2)
{
#if __CUDA_ARCH__ >= 900
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid  = threadIdx.x;
  const int cluster_id = (int)(blockIdx.x / CS);
  const int cell = cluster_id * CS + rank;      // global cell id owned by this block
  const int cnt  = g_count[cell];

#if BACKEND == 1
  __shared__ P4 smem_pos[MAXP_PER_CELL];        // DSM: reserve worst-case per cell
  P4* mypos = (P4*)cluster.map_shared_rank(smem_pos, rank);
#else
  P4* mypos = g_pos + (size_t)cell * MAXP_PER_CELL;   // global/RESC: no SMEM reserve
#endif

  // RESC: mark this cell's position buffer as cluster-coherent (once)
#if BACKEND == 2
  if (tid < 32) register_range_by_line_warp0(g_pos + (size_t)cell*MAXP_PER_CELL,
                                             (size_t)MAXP_PER_CELL*sizeof(P4));
  __syncwarp();
#endif
  cluster.sync();

  float acc = 0.f;
  for (int r = 0; r < repeat; ++r) {
    // ---- DRIFT / PRODUCE: integrate own particles, write new positions ----
    for (int i = tid; i < cnt; i += blockDim.x) {
      P4 p = g_pos[(size_t)cell*MAXP_PER_CELL + i];
      P4 v = g_vel[(size_t)cell*MAXP_PER_CELL + i];
      p.x += v.x*dt; p.y += v.y*dt; p.z += v.z*dt;   // drift
#if BACKEND == 1
      smem_pos[i] = p;                                // DSM: publish to own SMEM
#else
      write_pos(mypos, i, p);                          // global/RESC: publish (marked for RESC)
#endif
      g_pos[(size_t)cell*MAXP_PER_CELL + i] = p;      // keep global copy coherent
    }
    __syncthreads();
    cluster.sync();                                    // epoch barrier: produce -> consume

    // ---- FORCE / CONSUME: gather neighbor-cell positions, accumulate on home ----
    for (int i = tid; i < cnt; i += blockDim.x) {
      P4 pi = g_pos[(size_t)cell*MAXP_PER_CELL + i];
      float fx=0.f, fy=0.f, fz=0.f;
      // 27-cell stencil; in-cluster neighbors read peer positions (RESC L1-to-L1)
      for (int nb = 0; nb < 27; ++nb) {
        int ncell = g_neigh[cell*27 + nb];
        if (ncell < 0) continue;
        int ncnt = g_count[ncell];
        const P4* npos;
#if BACKEND == 1
        // DSM: map peer rank's SMEM if in same cluster, else global
        int nrank = ncell - cluster_id*CS;
        if (nrank >=0 && nrank < CS) npos = (P4*)cluster.map_shared_rank(smem_pos, nrank);
        else npos = g_pos + (size_t)ncell*MAXP_PER_CELL;
#else
        npos = g_pos + (size_t)ncell*MAXP_PER_CELL;   // RESC: marked peer lines served L1-to-L1
#endif
        for (int j = 0; j < ncnt; ++j) {
          P4 pj = read_pos(npos, j);                   // <-- the reused shared-written read
          float dx=pi.x-pj.x, dy=pi.y-pj.y, dz=pi.z-pj.z;
          float r2 = dx*dx+dy*dy+dz*dz;
          if (r2 > 1e-6f && r2 < rc2) {
            float inv = rsqrtf(r2); float inv2=inv*inv; float inv6=inv2*inv2*inv2;
            float f = (48.f*inv6*inv6 - 24.f*inv6)*inv2;   // LJ force magnitude
            fx += f*dx; fy += f*dy; fz += f*dz;
          }
        }
      }
      // second half-kick (force -> velocity), accumulate on HOME particle only
      P4 v = g_vel[(size_t)cell*MAXP_PER_CELL + i];
      v.x += fx*dt; v.y += fy*dt; v.z += fz*dt;
      g_vel[(size_t)cell*MAXP_PER_CELL + i] = v;
      acc += fx+fy+fz;
    }
    cluster.sync();
  }
  if (tid==0) g_out[cell] = acc;
#endif
}

#include <cstdlib>
#include <cmath>
int main(int argc, char** argv){
  int cluster_size = CLUSTER_SIZE;
  int num_clusters = 64;
  int repeat = 4;
  int pbar = 32;                 // mean particles/cell
  for (int i=1;i<argc;i++){
    if(!strncmp(argv[i],"num_clusters=",13)) num_clusters=atoi(argv[i]+13);
    else if(!strncmp(argv[i],"repeat=",7)) repeat=atoi(argv[i]+7);
    else if(!strncmp(argv[i],"pbar=",5)) pbar=atoi(argv[i]+5);
  }
  int num_cells = num_clusters * cluster_size;
  size_t npos = (size_t)num_cells * MAXP_PER_CELL;
  P4 *d_pos,*d_vel,*d_out_p; int *d_count,*d_neigh; float *d_out;
  CUDA_CHECK(cudaMalloc(&d_pos, npos*sizeof(P4)));
  CUDA_CHECK(cudaMalloc(&d_vel, npos*sizeof(P4)));
  CUDA_CHECK(cudaMalloc(&d_count, num_cells*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_neigh, (size_t)num_cells*27*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, num_cells*sizeof(float)));
  // host init: random-ish positions, per-cell counts ~pbar (dynamic), in-cluster neighbors
  P4* h_pos=(P4*)malloc(npos*sizeof(P4)); P4* h_vel=(P4*)malloc(npos*sizeof(P4));
  int* h_count=(int*)malloc(num_cells*sizeof(int)); int* h_neigh=(int*)malloc((size_t)num_cells*27*sizeof(int));
  for(int c=0;c<num_cells;c++){
    unsigned s=(unsigned)(c*2654435761u); s^=s>>13;
    int cnt = pbar/2 + (int)(s % (unsigned)(pbar));   // dynamic count in [pbar/2, 3pbar/2)
    if(cnt>MAXP_PER_CELL) cnt=MAXP_PER_CELL;
    h_count[c]=cnt;
    for(int i=0;i<cnt;i++){ unsigned t=s+i*40503u; t^=t>>7;
      h_pos[(size_t)c*MAXP_PER_CELL+i]=(P4){ (float)(t%1000)*0.01f,(float)((t/1000)%1000)*0.01f,(float)((t/1000000)%1000)*0.01f,0};
      h_vel[(size_t)c*MAXP_PER_CELL+i]=(P4){0.001f,0.001f,0.001f,0}; }
    // neighbors: the other cells in this cluster (in-cluster) + self, pad -1
    int cl=c/cluster_size, base=cl*cluster_size, k=0;
    for(int r=0;r<cluster_size && k<27;r++){ h_neigh[c*27+(k++)]=base+r; }
    for(;k<27;k++) h_neigh[c*27+k]=-1;
  }
  CUDA_CHECK(cudaMemcpy(d_pos,h_pos,npos*sizeof(P4),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vel,h_vel,npos*sizeof(P4),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_count,h_count,num_cells*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_neigh,h_neigh,(size_t)num_cells*27*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out,0,num_cells*sizeof(float)));
  dim3 grid(cluster_size*num_clusters,1,1); dim3 block(128,1,1);
  md_step<CLUSTER_SIZE><<<grid,block>>>(d_pos,d_vel,d_count,d_neigh,d_out,num_clusters,repeat,0.001f,6.25f);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  float* h_out=(float*)malloc(num_cells*sizeof(float));
  CUDA_CHECK(cudaMemcpy(h_out,d_out,num_cells*sizeof(float),cudaMemcpyDeviceToHost));
  double sum=0; for(int c=0;c<num_cells;c++) sum+=h_out[c];
  printf("OUT_CHECKSUM sum=%.6f n=%d BACKEND=%d cluster=%d num_clusters=%d pbar=%d\n",sum,num_cells,BACKEND,cluster_size,num_clusters,pbar);
  return 0;
}
