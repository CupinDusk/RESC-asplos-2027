VOID -- DO NOT USE THESE RESULTS.

The R=1 abort-check of markfrac-reuse-prereg.md (SHA-256 2a1d772392e7d187...,
frozen 2026-09-03T17:29:08Z) FAILED, so under its own rule S4.1 the entire
R-family is void and no R>=2 run was launched.

  frozen anchor arm0@0 (markfrac.cu)  = 114,006 cycles
  this variant  arm0@0 at R=1         = 114,728 cycles   (+722)

Cause: markfrac_reuse.cu adds a single-trip for-loop around the consumer read
that nvcc does not peel (reuse is a runtime arg), costing ~4 instructions per
consumer-thread execution = +108,544 instructions. Memory behaviour is
UNCHANGED (identical checksum, identical load/store counts) -- the discrepancy
is pure loop-control overhead, but at 722 cycles it is 79% of the 912-913 cycle
signal the experiment exists to measure, and it is asymmetric across arms.

Full analysis: resc-asplos-2027/markfrac-reuse-abort-report.md
Fix requires a NEW frozen pre-registration (compile-time R). Until then these
directories are evidence of a failed instrument check, not measurements.
