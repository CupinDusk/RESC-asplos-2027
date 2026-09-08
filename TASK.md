# 当前目标

将 `/legacy-research/owner/cudatest/RESC-c2c-wt` 的当前 RESC/ASPLOS 实验源码安全迁移到当前 GitHub 账号下的新私有仓库，使其他电脑可以从明确的 Git commit 继续构建和实验。

# 任务需求说明

- 保留已被预注册和论文结果引用的历史锚点 `d3cc7ad7`。
- 保存 `gpu-cache.cc` MD5 为 `dfb94a326b42f43e9d3e2dfdd1f47e2e` 的当前未提交机制状态。
- GitHub 作为跨机器代码同步点；新仓库默认设为 private。
- 同步机制源码、benchmark 源码、必要配置、运行/提取脚本、工具链 shim、小型结果摘要和迁移说明。
- 不把 9 GB 原始运行目录、构建产物、日志、PTX/PTXAS、实验二进制或备份文件加入新提交。

# 范围

## 包含

- 当前 Git 历史及 `d3cc7ad7` 锚点 tag。
- `src/` 中已跟踪的当前机制修改。
- `c2c_test/` 中可移植的源码、配置、脚本、`ptxas_shim` 和小型摘要。
- 当前 CUDA/编译器环境、关键 MD5、clone/build/验收说明。
- 新私有 GitHub 仓库、迁移分支和 tag。

## 不包含

- hLRC/VIPS-M backend 的设计与实现。
- 重跑或重新解释论文实验。
- 清理、删除或覆盖共享机器上的现有结果。
- 改写、rebase、squash 或删除已有 Git 历史。
- 将未发表代码推到现有公开仓库 `CupinDusk/RESC`。

# 约束

- 不执行 `git add -A`。
- 不使用 broad `kill`、`pkill` 或递归删除。
- 不覆盖已有 `.gitignore`，只做增量补充。
- `d3cc7ad7` 必须在远端可达，不能只在本机保留 tag。
- 当前工具链基准包含 `/usr/local/cuda-12.4/bin/nvcc` 12.4.99、`clustersim-py311` 和 `ptxas_shim`。
- 所有待提交文件必须通过 staged 文件清单和体积审查。

# 完成标准

- 新私有 GitHub 仓库已创建。
- 迁移分支已推送，远端 commit hash 与本地一致。
- `resc-published-anchor-d3cc7ad7` tag 已推送且指向 `d3cc7ad7`。
- 当前机制源码和必要实验配方已提交，生成物及原始日志未进入新提交。
- 迁移说明记录关键 MD5、工具链、构建入口、验收控制点和已知缺失项。
- 从远端只读查询能够看到迁移分支和 tag。
