p="resc_hybrid.cu"
s=open(p).read()
# Give the staging buffer its own compile-time size STAGE_ELEMS, independent of RESERVE_ELEMS.
# Default STAGE_ELEMS small so overflow is reachable; RESERVE_ELEMS still caps hotset (keep large).
s=s.replace(
"""      __shared__ float stage_buf[RESERVE_ELEMS];
      const int scount  = META_READ_S32(&meta->counts[rank]);
      const int soffset = META_READ_S32(&meta->offsets[rank]);
      const int sclamp  = scount > RESERVE_ELEMS ? RESERVE_ELEMS : scount;""",
"""      __shared__ float stage_buf[STAGE_ELEMS];
      const int scount  = META_READ_S32(&meta->counts[rank]);
      const int soffset = META_READ_S32(&meta->offsets[rank]);
      const int sclamp  = scount > STAGE_ELEMS ? STAGE_ELEMS : scount;""")
# add STAGE_ELEMS default near RESERVE_ELEMS defn
s=s.replace(
"""#ifndef RESERVE_ELEMS
#define RESERVE_ELEMS 4096""",
"""#ifndef STAGE_ELEMS
#define STAGE_ELEMS 512   /* hybrid SMEM staging buffer (elems); overflow beyond this re-fetches */
#endif
#ifndef RESERVE_ELEMS
#define RESERVE_ELEMS 4096""")
assert "stage_buf[STAGE_ELEMS]" in s and "define STAGE_ELEMS" in s
open(p,"w").write(s)
print("STAGE_ELEMS decoupled from RESERVE_ELEMS: OK")
