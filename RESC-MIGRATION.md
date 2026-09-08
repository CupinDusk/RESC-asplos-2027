# RESC 跨机器续跑入口

## Git 锚点

- 计划私有仓库：`https://github.com/CupinDusk/RESC-asplos-2027`
- 迁移分支：`resc-handoff-20260909`
- 已发表历史 tag：`resc-published-anchor-d3cc7ad7`
- tag 指向：`d3cc7ad77326c22ed497ed66d8ec5916de47f99c`
- 旧公开仓库 `https://github.com/CupinDusk/RESC` 不是本轮未发表代码的同步点。

新私有仓库创建并推送完成后，其他机器应从迁移分支开始：

```bash
git clone --branch resc-handoff-20260909 \
  https://github.com/CupinDusk/RESC-asplos-2027.git \
  /legacy-research/owner/cudatest/RESC-c2c-wt
cd /legacy-research/owner/cudatest/RESC-c2c-wt
git rev-parse HEAD
git rev-parse resc-published-anchor-d3cc7ad7
```

## 当前权威源码

迁移前校验值：

```text
src/gpgpu-sim/gpu-cache.cc  dfb94a326b42f43e9d3e2dfdd1f47e2e
c2c_test/maxunion.cu        52f64fb0809dccac49953acc82728cf0
build/lib/libcudart.so.12   c6d27da00388e9b667ea73e85f79ac0f
```

历史 `gpu-cache.cc` MD5 `1ee2dab9f932b25790a765967987bc3b` 是 2026-09-06 probe-port 计费补丁之前的状态，不是当前工作副本。

## 工具链

当前机器已确认：

```text
conda env     clustersim-py311
Python        3.11.13
GCC/G++       11.4.0
CMake         4.0.3
benchmark nvcc /usr/local/cuda-12.4/bin/nvcc
CUDA          12.4.99
```

论文已发表 benchmark 使用绝对路径 `/usr/local/cuda-12.4/bin/nvcc`。不要用默认 `nvcc` 替代；不同 NVCC 可能改变 PTX 和 `gpu_tot_sim_insn`。

其他关键约束：

- 使用 `-arch=sm_90`，不是 `sm_90a`。
- 编译前将 `c2c_test/ptxas_shim` 放在 `PATH` 最前。
- 编译后二进制需要执行 `patchelf --remove-rpath`。
- 不要在 NVCC 命令中加入 `-L<tree>/build/lib`；历史记录显示这会导致 RESC 臂静默失败。
- 当前脚本大量硬编码 `/legacy-research/owner/cudatest/RESC-c2c-wt`。首次迁移验收优先保持该路径，不要同时重构路径。
- `build_c2c.sh` 仍把 `/usr/local/cuda/bin` 放在 `PATH` 前面，而当前该 symlink 指向 CUDA 13.0；它不是已发表 benchmark 的 CUDA 12.4 编译命令。重新构建模拟器时必须记录实际 CMake/NVCC 选择并用控制点验收。

## 构建和运行入口

- 模拟器构建入口：`build_c2c.sh`
- benchmark 与实验入口：`c2c_test/run_*.sh`
- 当前唯一论文阻塞项：实现并预注册 hLRC/VIPS-M region-ownership 对照臂。
- hLRC backend 当前不存在；迁移完成前不要凭空构造运行命令。

## 新机器接受门

在新机器开展新实验前，先复现三个已发表控制点：

| 控制点 | 关键参数 | 期望 cycles | 期望 `gpu_tot_sim_insn` |
| --- | --- | ---: | ---: |
| `BIG_A2` | `2 32 4 2048 256`, knob 3, `RESC_GEN_RING=8`, `dl2 S:960` | 2,648,755 | 179,376,512 |
| `W3` | `RESC_GEN_RING=3` | 1,283,946 | 80,818,560 |
| `W8` | `RESC_GEN_RING=8` | 1,227,505 | 80,818,560 |

额外守恒检查：

- W3：`served=92`、`expired=197028`。
- W8：`served=197120`、`expired=0`。
- 两组都必须满足 `served + expired = 197120`。

判断规则：

- 指令数逐位相同：工具链可比，新结果可以与论文数据并列。
- 指令数相同但周期有小差：只报告差分比值，并记录机器差异。
- 指令数不同：不得与论文数字并列；先排查 NVCC/PTX 和模拟器构建漂移。

## 本次 Git 快照包含与排除

包含：

- 当前已跟踪机制源码修改；
- CUDA benchmark 源码；
- 顶层权威 `*.config`；
- Shell/Python 运行和提取脚本；
- `c2c_test/ptxas_shim/ptxas`；
- 小型 `*.txt`、`*.dat`、`*.csv` 结果摘要；
- `TASK.md`、`PLAN.md` 和本文。

排除：

- `build/` 和其他构建产物；
- `*.log`、`*.out`、`*.err` 原始输出；
- `*.ptx`、`*.ptxas`、`*.sass`；
- 实验二进制、共享库和静态库；
- `run_*`、`*_run`、`exit.code`；
- `*.bak*`、`*.MINE*` 备份。

## 尚未随代码树提供的证据

论文侧回答引用的 `*-prereg.md`、`*-result.md` 和论文 `data/*.dat` 当前不在此工作树，必须从论文/docflow 工作区另行导入或归档。缺少这些文件不影响保存当前源码，但在宣称完整 artifact/provenance 前必须补齐。

原始运行日志也不进入 Git。需要另做清单式压缩归档并记录 SHA-256；不要用只匹配单层目录的 glob，也不要删除原始文件。

论文侧记录还引用 pre-patch `libcudart.so` MD5 `0a27b8aac382e749c66569ff706014f0`。迁移检查在当前可见的 `/legacy-research/owner` 工作区没有找到该文件，只找到了 post-patch `c6d27da00388e9b667ea73e85f79ac0f`。前者必须从论文侧归档或其他机器另行找回；目前不能声称两个已发表 build 均已归档。

## 共享机器安全

- 不执行 `git add -A`。
- 不使用 broad `kill`、`pkill` 或递归清理。
- 不覆盖现有 `run.log`、`*.out` 或 `exit.code`。
- 运行实验前核对进程工作目录，并控制共享机器并发。
