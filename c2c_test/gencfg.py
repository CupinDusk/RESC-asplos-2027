import re
base=open("cfg_h20.config").read()
def scale_line(line, factor):
    m=re.match(r"^(-ptx_opcode_latency_\w+)\s+(.*)$", line)
    if not m: return line
    key,vals=m.group(1),m.group(2)
    nums=[str(max(1,int(round(int(x)*factor)))) for x in vals.split(",")]
    joined=",".join(nums)
    return key+" "+joined
factors={"1_0":1.0,"2_0":2.0,"3_0":3.0,"3_2":3.2,"4_0":4.0,"5_0":5.0}
for tag,f in factors.items():
    out=[]
    for line in base.splitlines():
        if line.startswith("-ptx_opcode_latency_"):
            out.append(scale_line(line,f))
        else:
            out.append(line)
    fn="cfg_cs"+tag+".config"
    open(fn,"w").write("\n".join(out)+"\n")
for tag in factors:
    for l in open("cfg_cs"+tag+".config"):
        if l.startswith("-ptx_opcode_latency_fp"):
            print("cs"+tag+": "+l.strip())
