# 🛠️ Benchmarking + Profiling

1. **Memory profiling with Memray**

    Ray supports memory profiling through `memray`,
    which is included in the server requirements. Each model
    that is executed should produce its own `$MODEL_NAME_mem_profile.bin`
    file that can be downloaded from the Ray dashboard for
    offline visualization. After starting the GPU server and running a few inference requests, pull up the Ray dashboard at `0.0.0.0:8265`, then go to
    `logs` -> `$NODE_NAME` -> `$MODEL_NAME_mem_profile.bin`
    to download the memory profile.

    Then run
    ```bash
    memray flamegraph $MODEL_NAME_mem_profile.bin
    ```
