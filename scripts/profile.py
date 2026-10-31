"""M5 profiling helpers. GPU box only.

    # timeline: where does a decode step actually spend its time?
    nsys profile -o results/decode --force-overwrite true \
        python scripts/profile.py timeline --batch 1 --steps 50

    # kernel counters: is the decode kernel near the memory roofline?
    ncu --set full --kernel-name-base demangled -k _paged_attn_decode \
        -o results/decode_ncu python scripts/profile.py kernel --batch 1

    # no external profiler needed: the launch-overhead story in one table
    python scripts/profile.py breakdown

What to look for in the nsys timeline, and what each thing means:

  * GAPS between kernels at batch size 1. If the GPU is idle 40-60% of the timeline,
    you have a LAUNCH OVERHEAD problem, not a kernel problem. The fix is CUDA graphs
    (--cuda-graphs), and the before/after timeline is the single most demoable image
    in the whole project.
  * A fat block of CPU time before each step. That is prepare_inputs building Python
    lists; it should be a few hundred microseconds, not milliseconds.
