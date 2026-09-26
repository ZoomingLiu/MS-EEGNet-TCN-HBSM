# MS-EEGNet–TCN–HBSI

癫痫 EEG 研究仓库：连续信号预处理、上游模型输出的因果后处理与诊断可视化。
本仓库不是经过临床验证的医疗设备；旧状态分类概率不等于未来 60 分钟发作概率。

## 当前状态

- `src/hbsm_loso/`：保留本地已修改的 HSMM、BOCPD 和交付接口。本次仅调整入口及输出路径，不改变其数学逻辑。
- `src/preprocessing/`：从 GitHub 提交 `016f70f3d08553f41196af5b4921af2102de936f` 整理的旧版预处理。仍采用人为状态标签，不是新的事件预测实现。
- MS-EEGNet/TCN 训练 notebook 不在该远端提交中；用户微信目录中的原文件没有擅自移动或复制进来。
- `src/forecasting/`：新增“过去 15 分钟 → 未来 60 分钟事件风险”的双分支 EEGNet–TCN 实验，含索引、训练、推理与测试。见 [长窗口实验说明](docs/LONG_CONTEXT_FORECAST.md)。尚未完成真实患者训练及临床有效性验证。

## 目录

```text
configs/                 可追踪的配置模板（本机路径覆盖用 *.local.yaml）
src/
  eeg_pipeline/          公共路径解析
  preprocessing/         旧版 Stage 1–3、数据加载与校验
  hbsm_loso/             HSMM / BOCPD / HBSI
  forecasting/           长历史形态/功率融合与未来事件预测
  Preprocessing.py       旧 notebook 大写导入兼容层
tests/                   单元测试与合成数据回归测试
notebooks/               探索性 notebook；不是生产入口
scripts/                 一次性维护与论文处理脚本
data/
  raw/                   不可覆盖的原始 EDF（也可配置仓库外路径）
  interim/               因果滤波信号与阶段性元数据
  processed/             按协议版本保存的最终索引、标签、划分
  external/              上游交付包
  subject_*/             已有数据的兼容位置，本次不搬动数组
outputs/                 运行结果、模型输出；不提交二进制
reports/                 图、指标、论文及渲染产物
docs/                    审查、交接、目录说明与历史文档
```

详细迁移映射见 [仓库布局说明](docs/REPOSITORY_LAYOUT.md)。GitHub 旧 README 保存在 `docs/archive/`，其中旧运行命令不再是当前入口。

## 安装

推荐新建虚拟环境，在仓库根目录执行：

```bash
python -m pip install -e ".[dev,preprocessing]"
# 需要数据加载器 / PyTorch 时另装 training extra；不会自动安装 GPU 专用版本。
python -m pip install -e ".[training]"
python -m pytest
```

只运行 HBSI 可使用 `python -m pip install -e .`，只依赖 NumPy。
运行长窗口实验使用 `python -m pip install -e ".[forecast,dev]"`。
当前提供依赖下限而非经全平台认证的锁文件；正式实验应另存环境版本。

## HBSI 与可视化

```bash
hbsi-run --data-dir data --output outputs/hbsi/run_001/results.pkl
hbsi-plot --input outputs/hbsi/run_001/results.pkl --data-dir data --output-dir reports/figures/run_001 --subjects subject_01
```

也可以使用 `python -m hbsm_loso.hbsi_main_loso` / `python -m hbsm_loso.hbsi_visualization`。
默认输出为 `outputs/hbsi/hbsi_loso_output_final.pkl`，已存在时需选择新路径或明确传入 `--overwrite`。
**pickle 只读取可信本地文件。** `--help` 不会加载数据或启动全量处理。

## 旧版预处理（复现实验，不是新预测协议）

先复制 `configs/preprocessing_legacy.yaml` 为 `configs/preprocessing.local.yaml`，设置真实 EDF 和信号目录。
配置中的相对数据路径相对于 YAML 文件；`CHBMIT_ROOT` / `EEG_SIGNAL_ROOT` 可覆盖路径，建议使用绝对路径。
运行前阅读审查报告中的时间轴、单位和缺失标注风险；不要把 Stage 1 输出指向已有冻结数据目录。

```bash
eeg-stage1 --config configs/preprocessing.local.yaml --output data/interim/stage1
eeg-stage2 --config configs/preprocessing.local.yaml --existing-metadata data/interim/stage1/metadata_all.parquet --output data/interim/stage2
eeg-stage3 --config configs/preprocessing.local.yaml --stage2-dir data/interim/stage2 --output data/processed/legacy_v2_2
eeg-validate --data-dir data/processed/legacy_v2_2 --baseline-dir data/interim/stage2
```

不安装命令行入口时，可用 `python -m preprocessing.stage1_edf_to_signals` 等模块形式。
包的默认目录不依赖启动位置；显式 CLI 相对路径仍按 shell 当前目录解释。
非 editable 安装时可用 `EEG_PROJECT_ROOT` 指定项目根目录。

## 实验边界

不随机拆分重叠窗口；LOSO 按实际患者分组；归一化只拟合训练折。
新预测任务通过 `forecast-prepare` 构建独立的未来事件标签和观察掩码，不能复用旧 `prediction_target`。它需要经核验的连续波形目录和真实发作事件表，不接受现有 TCN risk/embedding 代替 EEG。
既有结果仅移动保存，没有重新训练或宣称准确度提升。
