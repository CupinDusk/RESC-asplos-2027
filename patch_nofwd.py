f="src/gpgpu-sim/gpu-cache.cc"
s=open(f).read()
orig=s

a1='int g_c2c_noprobe       = 0;      // C cold-only control: CCOP present, never probes'
assert s.count(a1)==1, "a1 %d"%s.count(a1)
s=s.replace(a1, a1+'\nint g_resc_no_fwd       = 0;      // faithful-hLRC arm: disable marked-line cross-SM L1-to-L1 serve (RESC_NO_FWD)')

a2='  const char *dbg = std::getenv("RESC_C2C_DEBUG"); g_c2c_debug = dbg ? atoi(dbg) : 0;'
assert s.count(a2)==1, "a2 %d"%s.count(a2)
s=s.replace(a2, a2+'\n  const char *nf = std::getenv("RESC_NO_FWD"); if (nf) g_resc_no_fwd = atoi(nf);\n  if (g_resc_no_fwd) printf("\\n[RESC_NO_FWD] faithful-hLRC: marked-line cross-SM L1-to-L1 serve DISABLED (consumer refetches L2); L1-caching+self-invalidation intact\\n");')

a3='  (void)events;\n  if (!m_owner) return false;\n  if (mf->get_access_type() != GLOBAL_ACC_R) {'
assert s.count(a3)==1, "a3 %d"%s.count(a3)
s=s.replace(a3, '  (void)events;\n  extern int g_resc_no_fwd; if (g_resc_no_fwd) return false; // faithful-hLRC: no cross-SM forward -> L2 refetch\n  if (!m_owner) return false;\n  if (mf->get_access_type() != GLOBAL_ACC_R) {', 1)

# try_cluster_read_share: insert guard right after its opening brace
a4='bool l1_cache::try_cluster_read_share(new_addr_type addr, mem_fetch *mf,'
assert s.count(a4)==1, "a4 %d"%s.count(a4)
idx=s.index(a4)
brace=s.index('{', idx)
eol=s.index('\n', brace)
s=s[:eol+1]+'  extern int g_resc_no_fwd; if (g_resc_no_fwd) return false; // faithful-hLRC: no cross-SM forward\n'+s[eol+1:]

open(f,'w').write(s)
print("delta chars:", len(s)-len(orig))
print("g_resc_no_fwd occurrences:", s.count("g_resc_no_fwd"))
