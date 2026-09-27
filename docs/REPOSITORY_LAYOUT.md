# 仓库布局与迁移说明

日期：2026-09-26。

## 来源与范围

本地起点为 `1742144`，包含用户尚未提交的 HBSI 修改及旧版文件删除。
远端 main 检查快照为 `016f70f3d08553f41196af5b4921af2102de936f`。
本次没有执行 pull/merge/reset，没有恢复用户已删除的 `hbsm_v0`，也未引入远端 `hbsm_v1` 覆盖当前实现。
仅引入该远端快照的预处理源码、配置、原测试和 README 存档；调整包导入与路径默认值。

## 迁移映射

| 原位置 | 新位置 |
|---|---|
| GitHub 根目录的 Stage 1–3、`_utils.py`、`dataset.py`、`validate.py`、`export_member4.py` | `src/preprocessing/` |
| GitHub `config.yaml` | `configs/preprocessing_legacy.yaml` |
| GitHub `test_utils.py` | `tests/test_preprocessing_legacy.py` |
| `src/MEMBER2_HBSI_DATA_REQUEST.md` | `docs/handoffs/` |
| `src/UPSTREAM_MS_EEGNET_TCN_AUDIT_REPORT.md` | `docs/audits/` |
| `src/hbsi_loso_output*.pkl` | `outputs/hbsi/` |
| `src/hbsm_loso/hbsi_loso_output_final.pkl` | `outputs/hbsi/package_run/`（不覆盖同名结果） |
| `src/hbsi_final_visualizations/` | `reports/figures/hbsi/` |
| `src/hbsi_final_quality_report.csv` | `reports/metrics/` |
| `src/paper-*.docx`、论文 PDF 和渲染目录 | `reports/manuscript/` |
| `src/edit_paper_hbsi.py` | `scripts/edit_paper_hbsi.py` |
| 根目录 `member3_output128_causal_full_timeline/` | `data/external/member3_output128_causal_full_timeline/` |

已有 `data/subject_*`、`data/train`、`data/validation`、`data/loso_result` 不移动、不删除。
这不是推荐的新布局，而是为了保留既有索引和绝对路径的兼容位置；新产物按新目录写入。
用户的 `.vscode` 设置保留原处并忽略提交。旧审查报告内容作为历史记录保留，其中旧绝对路径须结合此表理解。

## 可恢复性

旧根 README 与 .gitignore 分别存入 `docs/archive/README.local-before-2026-09-26.md` 与 `gitignore.local-before-2026-09-26.txt`。
维护脚本默认 dry-run，应用前验证全部目标在明确指定的仓库内、目标不冲突、路径无目录链接。
文件移动校验 SHA256；目录移动校验文件清单和长度。迁移日志保存在 `reports/reorganization_20260926.json`。
需要回退目录移动时按日志反向逐项移动并先检查目标冲突；不要使用 `git reset --hard`，因为原本就有未提交修改。

## 代码约定

- 源码不写大型输出到 `src/`；模型和结果使用每次运行独立目录。
- 使用 editable 安装或包入口，不依赖在某个子目录中启动 Python。
- `preprocessing` 为规范导入名；`Preprocessing.py` 保留旧 notebook 的大写包导入。
- `hbsm_loso` 不改名，以减少旧 pickle 类路径不兼容风险。
- 原算法及原标签规则保留为 legacy；新 forecast 协议需独立配置、标签和测试。
- 原始 EDF 和已冻结信号只读复用；配置、代码和文档提交 Git，数据与产物不提交。
- 忽略规则不会自动移除已经被 Git 跟踪的文件；这次没有擅自改动暂存区。

## 发布状态

2026-09-26 的目录整理当时仅在本地完成。2026-09-27 的长窗口试验发布使用独立分支 `codex/long-context-forecast`，以核对过的远端 main 为父提交，不强推、不改变 main。
分支包含源码、配置、测试与必要说明，不上传本地数据、论文、模型或运行产物；上文迁移日志与本地存档不一定随分支发布。远端已有的 `hbsm_v0` / `hbsm_v1` 保留作历史参考，不恢复到用户已删除它们的本地工作区。
当前本地工作区与远端历史不同且含既有未提交改动，因此远端分支发布不等于已在本地切换分支；不要直接 reset 或覆盖工作区。
