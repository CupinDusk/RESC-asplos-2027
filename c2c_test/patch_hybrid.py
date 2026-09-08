import io
p="resc_hybrid.cu"
s=open(p).read()
s=s.replace(
"""#elif BACKEND == 0
  #define READ_S32(p)  ld_cg_s32(p)
  #define READ_F32(p)  ld_cg_f32(p)
  #define WRITE_S32(p,v) st_cg_s32(p,v)
  #define WRITE_F32(p,v) st_cg_f32(p,v)
#endif""",
"""#elif BACKEND == 0
  #define READ_S32(p)  ld_cg_s32(p)
  #define READ_F32(p)  ld_cg_f32(p)
  #define WRITE_S32(p,v) st_cg_s32(p,v)
  #define WRITE_F32(p,v) st_cg_f32(p,v)
#elif BACKEND == 3
  #define READ_S32(p)  ld_cg_s32(p)
  #define READ_F32(p)  ld_cg_f32(p)
  #define WRITE_S32(p,v) st_cg_s32(p,v)
  #define WRITE_F32(p,v) st_cg_f32(p,v)
#endif""")
old="""      float acc = 0.f;
      for (int pass = 0; pass < reuse_passes; ++pass) {
        const int count  = META_READ_S32(&meta->counts[rank]);
        const int offset = META_READ_S32(&meta->offsets[rank]);
        for (int i = tid; i < count; i += blockDim.x) {
          acc += PAYLOAD_READ_F32(&payload[offset + i]);
        }
      }"""
new="""      float acc = 0.f;
#if BACKEND == 3
      __shared__ float stage_buf[RESERVE_ELEMS];
      const int scount  = META_READ_S32(&meta->counts[rank]);
      const int soffset = META_READ_S32(&meta->offsets[rank]);
      const int sclamp  = scount > RESERVE_ELEMS ? RESERVE_ELEMS : scount;
      for (int i = tid; i < sclamp; i += blockDim.x) {
        stage_buf[i] = PAYLOAD_READ_F32(&payload[soffset + i]);
      }
      __syncthreads();
      for (int pass = 0; pass < reuse_passes; ++pass) {
        for (int i = tid; i < sclamp; i += blockDim.x) acc += stage_buf[i];
        for (int i = sclamp + tid; i < scount; i += blockDim.x) acc += PAYLOAD_READ_F32(&payload[soffset + i]);
      }
#else
      for (int pass = 0; pass < reuse_passes; ++pass) {
        const int count  = META_READ_S32(&meta->counts[rank]);
        const int offset = META_READ_S32(&meta->offsets[rank]);
        for (int i = tid; i < count; i += blockDim.x) {
          acc += PAYLOAD_READ_F32(&payload[offset + i]);
        }
      }
#endif"""
assert old in s, "consumer loop not found"
s=s.replace(old,new)
open(p,"w").write(s)
print("BACKEND==3 branch added:", "#elif BACKEND == 3" in s)
print("staging loop added:", "stage_buf" in s)
