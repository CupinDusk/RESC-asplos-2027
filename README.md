# RESC: Region-and-Epoch-Scoped Coherence

This repository is the public research implementation for **RESC: Region-and-Epoch-Scoped Coherence for Clustered GPUs**. It extends ClusterSim/GPGPU-Sim with communication-scoped local coherence, generation-qualified sharing, and the cross-cluster owner predictor (CCOP).

RESC maintains declared communication data during its live epoch within a thread-block cluster. Across clusters, a bounded owner-hint table selects a candidate peer, with generation checks and L2 completion. The prototype uses simulator-side owner records for learning and validation; physical-cache and owner-record confirmations are reported separately.

## Source identity and artifact map

The manuscript's implementation snapshot is [9f22b57b374b605bffb5a27fe1d80181ae0beb6c](https://github.com/CupinDusk/RESC-asplos-2027/tree/9f22b57b374b605bffb5a27fe1d80181ae0beb6c). The historical repository name preserves the project's original venue plan.

| Material | Location |
|---|---|
| RESC cache mechanisms, generation tracking, and CCOP | [gpu-cache.cc](src/gpgpu-sim/gpu-cache.cc), [gpu-cache.h](src/gpgpu-sim/gpu-cache.h) |
| Corrected local H20 configuration (78 SMs, 60-MiB L2) | [cfg_h20_l2_60mb.config](c2c_test/cfg_h20_l2_60mb.config) |
| CUDA benchmarks and per-cell configurations | [c2c_test](c2c_test/) |
| Cross-cluster benchmark | [resc_c2c.cu](c2c_test/resc_c2c.cu) |
| Archived raw-log hashes and gzip paths | [RESULTS-MANIFEST.tsv](c2c_test/RESULTS-MANIFEST.tsv) |

Each experiment is bound to its own configuration. The retained CCOP profiles use 128 SMs and 4-MiB L2; the local H20 study uses the corrected 78-SM/60-MiB configuration. Archived logs are available as `run.log.gz` beside each cell's configuration. The manifest records both compressed and original-log SHA-256 values.

## Build and data use

The simulator requires a Linux/CUDA toolchain. Upstream build instructions follow below. Set `GPUSIM_CONFIG` to the configuration of the specific study being reproduced. Historical experiment scripts and paths document their original environment; adapt them to the chosen environment and verify a control cell before extending experiments.

The public release preserves the existing implementation and archived results. Fresh clean-environment simulator execution has a separate validation scope from source and log-hash verification.

## License and attribution

RESC-authored additions and modifications are released under the BSD 3-clause redistribution terms in [COPYRIGHT](COPYRIGHT). All upstream and third-party notices and their respective terms remain in force. Please also cite ClusterSim and GPGPU-Sim as described below.

---

# ClusterSim

**ClusterSim** is a cycle-accurate simulator for modeling modern GPUs executing CUDA programs.
It is built on top of [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) and extends its functionality to support new hardware and programming features.

---

## New features compared to GPGPU-Sim 4.0

* Group cores into GPCs with `gpgpu_n_cores_per_gpc` parameter
* New Special registers: `%cluster_ctaid`, `%cluster_ctarank`, `%cluster_nctaid`, `%cluster_nctarank`...
* New Instructions:  `mapa`, `barrier.cluster`
* New directives: `.explicitcluster`, `.maxclusterrank`, `.reqnctapercluster`
* New API calls: `cudaLaunchKernelExC`, `cudaMallocManaged`,
* Simulationmodel for the SM to SM interconnect

---

## ⚙️ Setup

### Dependencies

Install required packages:

```bash
sudo apt-get install -y build-essential xutils-dev bison zlib1g-dev flex \
    libglu1-mesa-dev git cmake ninja-build clang clang-format
```

Make sure the CUDA toolkit is in your `PATH`:

```bash
export PATH=$PATH:/usr/local/cuda/bin
```

---

### 🔨 Build

The simulator builds as a shared library. CUDA applications linked against `cudart` (as a shared library) can be run on the simulator by adjusting the dynamic link loader path.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

---

### ▶️ Run

By default, ClusterSim uses the configuration file at
`configs/tested-cfgs/SM90_H100/gpgpusim.config`.

You can change this via the `GPUSIM_CONFIG` environment variable.

```bash
# If not already in rpath:
source enable_simulator.sh

# Verify CUDA program links to simulator’s libcudart
ldd build/bin/network

# Run a CUDA program
./build/bin/network

# Change GPU config
export GPUSIM_CONFIG=<PATH_TO_CONFIG>/gpgpusim.config
./build/bin/network
```



### Notes

If you use ClusterSim in your research please cite

```bibtex
@article{ClusterSim,
    title={ClusterSim: Modeling Thread Block Clusters in Hopper GPUs},
    url = {https://tore.tuhh.de/handle/11420/57345},
    doi = {10.15480/882.15858},
    author={Lühnen, Tim and Behera, Jyotirman and Tripathy, Devashree and Lal, Sohan},
    year={2025}
}
```

Also cite the original [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) project.

