# 当前理解

当前权威工作树位于 `/legacy-research/owner/cudatest/RESC-c2c-wt`。迁移必须保留历史锚点 `d3cc7ad7`，并将当前机制源码、可重建实验所需配方、实验目录和已有结果提交到当前 GitHub 账号下的新 private 仓库。现有公开 `origin` 不接收本轮未发表代码。

实测 `c2c_test/` 为 8.1 GiB / 10,804 个文件，其中 2,010 个实验原始 `.log` 合计 7.766 GiB，15 个单文件超过 100 MiB；另有 21 个 `bin/` 下的空编译日志，按生成物排除。可直提交的配置、源码、脚本和非 log 结果约 34.2 MiB。因此原始 log 保留在本地，Git 中保存确定性 gzip 副本和完整性 manifest。

# 需要先阅读 / 复核的文件

- `TASK.md`：迁移范围、约束和完成标准。
- `.gitignore`：确认新增排除规则不会覆盖现有规则。
- `build_c2c.sh`：记录当前构建入口和硬编码路径。
- `c2c_test/run_lh.sh`、`c2c_test/run_app.sh`、`c2c_test/run_discrim.sh`：确认续跑依赖和验证指标。
- 当前 `git status`、`git diff --stat`、关键 MD5、CUDA 12.4 工具链和远端状态。
- `c2c_test/` 中文件数量、类型、总体积、大文件及正在运行的进程。

# 本轮计划修改的文件

- `.gitignore`：保持对未压缩原始日志、PTX/PTXAS/SASS、构建产物和备份的排除，同时允许实验结果中的 `*.out`、`*.err`、`exit.code` 和 `gpgpu_inst_stats.txt`。
- `TASK.md`：随迁移提交保存已确认的目标、范围和完成条件。
- `PLAN.md`：保存本轮迁移步骤、风险和验证路径。
- `RESC-MIGRATION.md`：新增跨机器入口，记录仓库/分支/tag、关键 MD5、环境、构建方式、验收控制点和已知缺失证据。
- `c2c_test/archive_results_for_git.sh`：新增确定性压缩原始 log 并生成完整性 manifest 的工具。
- `c2c_test/RESULTS-MANIFEST.tsv`：记录每个原始 log 的路径、字节数、SHA-256、gzip 路径和压缩体积。
- 范围内的 `c2c_test/**/*.log.gz`：确定性压缩的原始实验 log，原 `.log` 不改动；`bin/` 中的编译日志排除。
- `c2c_test/**/*.{config,txt,dat,csv,out,err,cu,cpp,py,sh,md,bib}` 与 `c2c_test/**/exit.code`：精确暂存实验说明、配置、源码、运行脚本、状态和结果；已跟踪文件不因此被批量改写。

不计划修改原始 `.log`、PTX/PTXAS/SASS、二进制、库、对象文件、备份、PID 或运行中标记。

既有机制源码、benchmark、配置、脚本及摘要只做审查和选择性暂存，不在本轮改写其内容。

# 最小实施方案

## Milestone 1：本地迁移锚点（已完成）

- 从 `d3cc7ad7` 所在历史创建 `resc-handoff-20260909` 分支。
- 创建 `resc-published-anchor-d3cc7ad7` tag。
- 验收：分支祖先包含 `d3cc7ad7`，tag 精确指向该提交，当前工作区修改保持不丢失。

## Milestone 2：code-and-recipe 快照（已完成）

- 增量更新 `.gitignore`。
- 新增 `RESC-MIGRATION.md`。
- 精确暂存：已跟踪机制源码、必要 benchmark 源码、顶层权威配置、运行/提取脚本、Python 工具、`ptxas_shim`、小型结果摘要及迁移文档。
- 不暂存：构建产物、日志、PTX/PTXAS、无扩展名实验二进制、运行目录、备份和退出码。
- 验收：检查 staged 文件清单、文件数、总 blob 体积、最大 blob、`git diff --cached --check`，并确认关键 MD5 未漂移。

## Milestone 3：私有 GitHub 同步点（已完成）

- 使用当前已登录 GitHub 账号创建新的 private 仓库 `RESC-asplos-2027`；若名称已占用，选择不覆盖现有仓库的新名称。
- 新增独立 remote，不修改现有公开 `origin`。
- 提交并推送迁移分支与锚点 tag。
- 验收：GitHub 报告仓库为 private；远端分支 commit 与本地一致；远端 tag 指向 `d3cc7ad7`。

## Milestone 4：实验目录与结果快照（已完成）

- 修改文件：`TASK.md`、`PLAN.md`、`.gitignore`、`RESC-MIGRATION.md`、`c2c_test/archive_results_for_git.sh`、`c2c_test/RESULTS-MANIFEST.tsv`、白名单结果文件和 `c2c_test/**/*.log.gz`。
- 对原始 `.log` 生成确定性 gzip 副本，不删除、截断或覆盖原文件。
- 用明确的文件类型白名单暂存实验说明/配置/结果，不执行 `git add -A`。
- 验收：manifest 与范围内的实验原 log 一一对应；全部 gzip 通过 `gzip -t`；暂存区无编译生成物/超限 blob/凭据；远端分支 hash 与本地一致。

# 风险点

- `d3cc7ad7` 自身包含大量历史日志；推送 descendant 必然携带该可达历史，不能只推子提交而省略 parent。
- 当前 `.gitignore` 不充分，宽泛 `git add` 会误收数千个生成文件和 `src/*.bak_*`。
- 预注册和论文 `data/*.dat` 证据文件当前不在此工作树；迁移说明必须明确记录缺失，不能伪装为已同步。
- 运行脚本硬编码绝对路径；本轮不同时改动路径和迁移状态，其他电脑优先使用同一路径或按说明处理。
- GitHub 外部写入只针对新 private 仓库；现有公开 `origin` 保持不变。
- 未压缩 log 中有 15 个超过 GitHub 100 MiB 硬限，必须压缩后逐文件复查；若仍有超限文件，不得强推或开启 LFS，需要重新调整归档粒度。
- 大量实验目录包含无扩展名 ELF 二进制，仅靠文件名不能过滤；暂存后必须用 blob magic 再检查。

# 最小验证路径

1. `git status --short` 和 `git diff --cached --name-status` 审查暂存范围。
2. `git diff --cached --check` 检查补丁格式。
3. 校验 `gpu-cache.cc`、`maxunion.cu` 和当前 `libcudart.so.12` MD5。
4. 校验 `/usr/local/cuda-12.4/bin/nvcc` 为 CUDA 12.4.99。
5. 校验迁移分支包含 `d3cc7ad7`，锚点 tag 精确解析为 `d3cc7ad7`。
6. 创建并推送后，通过 GitHub API/CLI 和 `git ls-remote` 验证 private 可见性、分支和 tag。
7. 校验压缩 log 数量、manifest 数量、SHA-256 与 `gzip -t`，并检查压缩后最大文件。
8. 对 staged 文件执行类型/体积/凭据审查，确认没有 PTX/PTXAS/SASS、ELF、库、对象文件或超 100 MiB blob。
9. 推送后用 `git ls-remote` 确认远端分支精确指向本地 HEAD。
