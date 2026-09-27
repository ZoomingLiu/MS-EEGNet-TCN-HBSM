# 长窗口风险预测实验 v1

实现日期：2026-09-27。

此分支新增可运行的网络、索引准备、训练和推理流程。它是研究试验，不是已证明有效的癫痫预测器。
旧 `preprocessing` 与 `hbsm_loso` 保留作历史对照；不修改微信目录中的原 notebook，也不覆盖已有数据、模型和结果。

## 1. 实现内容

- 输入：过去 15 分钟，22 通道、256 Hz 的连续、因果滤波后 EEG，物理单位 uV。
- 每分钟更新一次；局部窗口为不重叠的 5 秒窗，每分钟 12 个。
- 形态分支：沿用 `MS-EEGNet-6.ipynb` 的 EEGNet 式时间卷积、空间深度卷积及两段可分离卷积，输出每窗 16 维，不再附带局部状态分类损失。
- 为避免训练中跨窗口批统计和分块编码的耦合，将 BatchNorm 改为单窗 GroupNorm。因此**不是旧 checkpoint 的无损替换**，不能直接加载旧 notebook 权重。
- 每分钟对 12 个向量取均值、标准差、末窗向量，48→64 维。
- 第二分支从逐窗标准化前的波形计算 log RMS、log 线长、5 个频带 log 功率，再取分钟内均值、标准差和斜率，462→64→32 维。
- 拼接 96 维，经融合层到 64 维；4 个因果 TCN 残差块，k=3、每块两层、dilation=[1,2,4,8]。
- 理论感受野为 61 个分钟 token；输入长度超过感受野时配置校验拒绝运行。
- 风险头 64→32→1，训练使用 BCEWithLogitsLoss，监督只作用于最终端点。

这不是新发明的 EEGNet/TCN，而是适配事件目标、长历史和尺度信息保留的实验结构。

## 2. 安装与命令

在项目根目录、独立 Python 环境中：

```bash
python -m pip install -e ".[forecast,dev]"
python -m pytest

forecast-prepare --config configs/forecast_long_context.yaml --records data/processed/catalog/records.csv --events data/processed/catalog/events.csv --output data/processed/forecast_v1/index.json
forecast-train --config configs/forecast_long_context.yaml --index data/processed/forecast_v1/index.json --output-dir outputs/forecast/run_001
forecast-predict --checkpoint outputs/forecast/run_001/best.pt --index data/processed/forecast_v1/index.json --split test --output-dir outputs/forecast/run_001_test
```

对应模块入口为 `python -m forecasting.prepare`、`python -m forecasting.train`、`python -m forecasting.predict`。
GPU 版 PyTorch 请根据训练机器单独选择；CPU 可将配置 `device: cpu`。基础依赖安装不自动下载 EDF 或模型。
输出目录必须是新的；不会自动覆盖既有 checkpoint。Torch checkpoint 使用 `weights_only=True` 加载，仍应只读取可信来源。

## 3. 必需的两张输入表

### records.csv：连续信号目录，不是 TCN 输出目录

| 列 | 定义 |
|---|---|
| subject_id | 真实患者身份；chb01/chb21 必须映射为同一患者 |
| session_id | 一次采集会话的时间基准；不同住院/不同 case 不要盲目拼接 |
| record_id | 物理文件标识，同一患者/会话内唯一 |
| signal_path | 连续 `[C,N]` float32/float64 NPY；相对路径相对于该 CSV |
| start_sec | 经过验证的、会话内文件起始秒数；不能缺失时假设 0 或一小时顺序 |
| sample_rate | 必须与配置一致，默认 256 |
| channel_names | JSON 字符串数组，顺序必须与配置严格一致 |
| signal_unit | 必须明确为 `uV`，不能通过数据幅度猜测 |
| causal_filter_verified | 明确 true/false；只有经核验的因果滤波信号允许进入此管线 |
| annotation_complete | 此文件区间内是否有完整发作标注；false 会导致覆盖它的未来标签未知 |
| split | train / validation / test，由预先固定的患者或时间划分给出 |

文件实际持续时间由 NPY 样本数 / fs 得到。准备阶段检查通道数、重复文件、重叠时间、单位和采样率；记录文件长度及修改时间，训练时发现源信号改变会报错。
这些核验字段是上游数据契约，**代码无法仅凭 NPY 证明物理单位或滤波算法正确**；不能为了通过校验而统一填 true。

### events.csv：真实发作事件

需要 `subject_id,session_id,onset_sec,offset_sec` 四列，时间与 records.csv 使用同一会话基准。
没有发作时用仅含表头的 CSV，而不是虚构零时间事件。发作起点必须落在记录覆盖区间，起止时间必须有限且 offset > onset。
数据集的发作前期/过渡期人工标签不进入新目标。

### 如何利用已有 Stage 1 元数据

可按 `(subject_uid,case_id,record_id,signal_path)` 分组导出候选记录：
`subject_uid→subject_id`、`case_id→session_id`，候选起点通常是 `absolute_start_sec-window_start_sec`。
必须检查同记录内推算值唯一，并与 summary/EDF 头核对；旧代码包含时间缺失回退，因此本实现**不自动把旧元数据视为已经核验的 catalog**。
发作时间需从原 summary 的 record-local onset/offset 加上已验证的 record start，不能从 labels.npy 反推。
当前本地 `data/subject_*` 只有上游 risk/embedding 等产物，不能替代连续波形和事件目录。

## 4. 标签、因果性与划分边界

在非发作、非预设发作后排除期的时刻 t：

```text
input = EEG[t - history, t)
target = 至少一个发作起点落入 (t, t + horizon]
```

- 输入历史必须完整，无跨 split、无未观察缺口、无 ictal 波形。
- 每份物理记录的前 `filter_warmup_seconds` 不进入历史输入；第一版采取保守做法，不推断其滤波器状态连续。
- 未来完整观察且无发作才标 0；完整观察且有发作标 1；随访不足或存在缺口标 -1，并设置 `target_observed=false`。
- 第一版即使已看到一次发作，只要未来区间不完整，也采用保守的未知标签策略；不是生存分析模型。
- `(t,t+H]` 的右端点也必须落在同 split 的已观察记录内，防止正好跨验证/测试边界的事件标签泄漏。
- 同 record 不能作为不同路径重复放进目录；同一 session 内记录不可重叠。相邻文件只有真实时间无缝、且符合上述条件时才能组合。
- 按时间划分时，历史与未来覆盖自动在 split 边界留出隔离区；LOSO 应按 subject_id 划分并单独准备每一折 catalog/config/run。
- 固定 horizon 不是声称生理 preictal 持续 60 分钟。

CHB-MIT 许多 EDF 之间有短缺口，这个严格基线可能丢弃大量未来一小时标签。先看 prepare 打印的各 split 标签计数与独立事件数。若几乎没有有效正负样本，应另行研究更短时域或删失建模，**不能把缺口默认为没有发作**。

## 5. 训练与防泄漏

形态分支在每个已结束 5 秒窗内标准化；第二分支先保留物理尺度，再使用仅训练集拟合的 FeatureScaler。
FeatureScaler 按训练端点包含的分钟特征统计，因此重叠历史会重复计权；没有用验证/测试患者拟合它。
默认自然采样、无类别权重；仅验证集 Brier score 早停，测试集不参与选择 checkpoint。
训练需要正负类都存在、验证集有可观察样本；小样本仅含一种类别时直接报错而不制造成功指标。
数据按索引 mmap 读取，编码器分块与可选 activation checkpoint 控制内存；没有把每个重叠 15 分钟 EEG 另存成大数组。

## 6. 消融与冻结实验

- 改 `branches` 为 morphology / power / both，保持同一索引与 TCN 宽度。
- 改 `history_minutes` 为 1 / 5 / 15 / 30 后必须重新 prepare；比较时按患者、会话、端点取共同可评估集合，而不是比较不同样本集合的裸指标。
- `encoder_checkpoint` 支持本实现保存的 checkpoint；默认 joint 训练。
- 冻结实验设 `freeze_encoder: true` 并指定 `encoder_checkpoint`；随机编码器不允许冻结后当作预训练模型。
- checkpoint 的训练索引签名必须与当前训练索引完全一致，拒绝使用其他 fold 的训练编码器。
- 正式 Frozen/Joint 对照应从同一个合规训练阶段的 encoder checkpoint 出发，另设相同训练预算；先 warm-up 的预算也要报告。不能把不同初始化直接当作单因素消融。

## 7. 产物与解释限制

训练输出：`config.json`、`scaler.json`、`history.json`、`best.pt`。
推理输出：`predictions.csv`、`metrics.json`。保留 subject/session/record、端点、horizon、target_observed、logit 和 risk。
推理包括输入有效但未来结果未知的端点；未知标签不参与评价。
报告 AP、ROC-AUC、Brier、相对训练基线的 Brier skill，以及辅助 Accuracy@0.5；并处理常数分数的并列排名。
所有 risk 明确标记 `probability_calibrated=false`，未经校准不能声称是可靠的临床概率。
本版尚未实现概率再校准、独立事件级报警统计、置信区间或真实患者性能验证，这些是后续评价工作。

**不要直接接旧 HBSI 的 2.5 秒步长管线。** 新输出为一分钟一次、事件风险语义，旧持续状态和 hazard 参数需按时间尺度重新设计；也不能伪装成旧 `[N,128]` TCN embedding 交付。

## 8. 验证范围

自动化测试覆盖：默认 `[1,15,12,22,1280]` 前向、各分支反向传播、TCN 前缀因果性、幅度尺度保留、8 个固定 token 样本过拟合、标签右边界、观察缺口、split 隔离、未知标签、checkpoint 来源，以及合成信号的 prepare→train→predict。
测试通过说明工程链路可运行，不等于证明真实 CHB-MIT 发作可预测。当前没有把原来的 0.685 Accuracy 替换为任何新性能声明。
