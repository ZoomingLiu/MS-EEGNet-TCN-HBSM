# 预处理审查：哪些需要改，哪些可以复用

审查日期：2026-09-26。远端快照：[016f70f3d08553f41196af5b4921af2102de936f](https://github.com/ZoomingLiu/MS-EEGNet-TCN-HBSM/tree/016f70f3d08553f41196af5b4921af2102de936f)。
本次整理引入的源码位于 `src/preprocessing/`；以下结论来自该快照，不假定本地所有历史数据都由这个版本生成。

## 结论

**需要修改任务标签和时间组织，但不应直接推倒全部波形预处理。**
若原有大容量数据确实是 Stage 1 保存的、单位正确且包含完整记录的连续滤波信号，可复用这些信号，重建元数据、未来事件标签、长历史索引与统计特征。若只保留上游 TCN risk / embedding，不能从中恢复波形或幅度信息。
本地目前看到的 `data/subject_*` 是 TCN 输出交付，不能据此证明那批约 70 GB 连续信号的质量和来源。需要核对真实 signal_root、审计报告和数据哈希。

## 1. 可以保留的部分

| 环节 | 代码证据 | 判断 |
|---|---|---|
| 因果带通滤波 | `_utils.py:apply_causal_bandpass` 使用 `butter(..., output='sos')` 和单向 `sosfilt` | 不需要仅因为预测尺度变长就重新滤波；不是 filtfilt |
| 连续信号保存 | `stage1_edf_to_signals.py` 保存 `filtered.astype(np.float32)`，尚未执行逐窗 z-score | 原始尺度可能仍然存在，可新增幅度/频带功率分支 |
| 5 秒窗口索引 | `config.yaml` 的 5 秒长度、2.5 秒步长 | 仍可作为局部编码单元；不必要求每窗给出可靠状态判断 |
| 患者身份映射 | `subject_mapping` 已将 chb01/chb21 合并到 subject_01 | 保留；本次并未发现这里把同一患者拆开 |
| 折内归一化 | `stage3_sequence_build.py:generate_loso_folds` 排除 test/validation 患者计算统计；`dataset.py` 使用固定 mean/std | 不应笼统描述为所有代码都逐窗归一化；模型 notebook 是另一条实现路径 |

参考原文件：[滤波与工具函数](https://github.com/ZoomingLiu/MS-EEGNet-TCN-HBSM/blob/016f70f3d08553f41196af5b4921af2102de936f/_utils.py#L754)、[连续信号保存](https://github.com/ZoomingLiu/MS-EEGNet-TCN-HBSM/blob/016f70f3d08553f41196af5b4921af2102de936f/stage1_edf_to_signals.py#L244)、[固定统计归一化](https://github.com/ZoomingLiu/MS-EEGNet-TCN-HBSM/blob/016f70f3d08553f41196af5b4921af2102de936f/dataset.py#L136)。

## 2. 适配未来事件预测必须修改

### P1：新标签不能由旧状态标签映射得到

`compute_prediction_target` 当前把 INTERICTAL→0、PREICTAL/TRANSITION→1，其余→-1。
Stage 3 取末窗口的这个 target；因此它仍是人为定义状态的分类，不是未来 60 分钟发作事件预测。

新增独立的 endpoint 表：

```text
subject_uid, case_id, record_id, session_id
endpoint_abs_sec, history_start_abs_sec, forecast_end_abs_sec
history_observed_fraction, future_observed_fraction
input_quality_mask, target_observed_mask, target_future_60min
next_seizure_onset_abs_sec, split_id, protocol_version
```

目标为 `(t, t+3600]` 内是否出现真实标注的发作起点；未来不足 60 分钟或观察有缺口时不能默认为负类。第一版可统一只纳入未来完整可观察的端点；之后若使用删失模型，必须另行定义。
标签允许离线读取未来事件，输入特征及运行时质量控制不得读取未来。
保留 UNKNOWN 区域的 EEG 作为候选输入，而不是依据旧人为状态标签永久删除它们。

### P1：重建分钟级、长历史索引

当前 `sequence.length=24`，5 秒窗 / 2.5 秒步长只覆盖 `5 + 23×2.5 = 62.5` 秒。
Stage 3 还明确限制同一 record。新任务应每分钟生成端点，索引过去 15 分钟的窗口/信号，再分层编码；不要把连续 15 分钟复制成大量重叠的大数组。
跨 record 只允许依据真实时间关系构建；不能忽略缺口把两个文件直接拼接。模型历史与标签观察覆盖是两种不同掩码。
不同历史长度消融应使用共同可评估端点，防止输入长度变化同时改变测试事件集合。

### P1：双分支的数据出口

形态分支可按窗标准化；幅度分支读取标准化前的连续滤波信号，计算 log RMS、线长、频带功率及分钟趋势。
统计特征的 scaler 只在训练折拟合。若再次对每个窗口独立去均值/单位方差，将抹掉希望保留的尺度变化。
0.5 Hz 高通已去掉的极低频/DC 成分无法恢复；研究表述应是幅度包络/功率变化，而非声称恢复了原始基线漂移。

### P1：新划分与评估协议

旧 2:1 采样是分类训练策略，不是自然发作概率。验证/测试保留自然时间轴，重采样或加权训练后需单独验证概率校准。
按时间划分时隔离历史窗口与未来标签覆盖，避免同一发作和高度重叠窗口跨训练/测试；跨患者研究按实际 subject_uid 分组。
事件检出率、报警时间占比、误报率、提前时间和概率校准取代“只报 Accuracy”。

## 3. 已发现的旧实现风险：新任务前必须处理/核查

### P1：缺失时间/标注时的危险回退

`build_patient_timeline` 在时间信息缺失时把每文件假设成 3600 秒顺序拼接；Stage 1 在找不到 timeline_entry 时使用起点 0、终点 3600、空 seizures。
这会把“未知时间/未知标注”伪装成明确的无发作区间，尤其不适用于未来一小时标签。
建议：使用经验证的 EDF 时间头、真实样本数及标注来源修复；无法验证则标记 unknown/exclude，并输出原因，不能猜测无发作。
时间基准至少要包含 case/session；同一患者的不同住院记录不能只按 subject_uid 拼成同一连续时间轴。

### P1：单位转换由整份文件的幅度决定

`read_edf_with_mne` 根据全记录最大绝对值是否小于 0.1 决定是否乘 1e6，而两条分支都标记为 uV。
这既可能在异常高幅值/混合通道记录上错误标记尺度，又使较早信号的转换分支取决于后续样本。
建议按读取 API 的单位约定和通道类型显式取 EEG 微伏数据，例如经通道核验后使用 `get_data(..., units='uV')`，不要用全记录幅度猜单位。
MNE 默认数据单位及 units 参数见 [官方 Raw.get_data 文档](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data)。
这属于条件风险，不代表所有已生成文件都错了；先抽查单位/峰值/头信息，再决定是否仅重跑受影响记录。

### P2：配置与实际行为不完全一致

- `build_patient_timeline` 内部将 gap_threshold 写死为 30 秒，未接收配置中的阈值。
- `apply_causal_bandpass` 的 warmup 写死为 2 秒，修改 YAML 未必改变它。
- 短缺口归为同一个 block，不代表缺口内已经观察到了信号或发作。
- `fit_normalization_streaming` 仍会把整份信号转换为 float64 并分配累计和数组；长记录有内存压力。
- 归一化函数读取某个信号失败时会跳过；应审计失败列表，不能静默改变拟合样本。
- `dataset.py` 对字符串 window_ids 使用 eval；应单独改成安全解析并验证类型。

本次目录重构没有悄悄改动上述科学处理逻辑；这些问题保留在 legacy 包中以便复现，并在这里列为下一阶段修复清单。

### 旧测试与实现不一致（本次修正测试）

远端 `test_window_boundary` 断言 50%/50% 重叠不是 boundary，但 `_utils.py` 的 V2.1 规则明确要求至少 80% 重叠；Stage 2 也使用该规则。
本次保留算法，修正旧测试预期，并补充 79% / 80% / 81% 阈值测试与滤波未来扰动不影响历史的回归测试。这不是通过修改算法去迎合测试。

## 4. 最小重处理策略

1. 找到现有连续信号与原元数据，核验版本、单位、通道、采样率、时间覆盖。
2. 先统计各患者在“15 分钟历史 + 60 分钟未来”的严格覆盖条件下还剩多少端点和独立发作。
3. 复用通过验证的信号；重新生成事件/观察区间表和未来标签。
4. 重新生成分钟索引及幅度/功率缓存，不重新复制整个 EEG。
5. 按新划分重新拟合归一化统计，重训/重新导出对应任务的 embedding；旧标签训练的 TCN risk 不等于新标签风险。
6. 用小样本工程诊断、线性基线、短/长历史和双分支消融验证。

若严格覆盖后几乎没有样本，需显式调整预测时域或引入删失建模，不得把缺口当成负例以凑样本。

## 5. 本次已经做与尚未做

已做：检查远端源码；包化、路径整理、配置模板、入口及测试组织；移动本地产物并保留恢复映射。
未做：执行 EDF 重处理、改写旧标签、生成新的 forecast 数据、训练新网络或重新计算临床预测性能。
本次组织层改动通过测试并不等同于旧预处理所有科学假设已经正确。
