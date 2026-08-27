import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_data_root(start_dir: Path) -> Path:
    candidates = [
        start_dir / 'data' / 'data',
        start_dir / 'data',
        start_dir / 'member3_output128_causal' / 'data',
        Path('data/data'),
        Path('data'),
        Path('.')
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            subdirs = [d for d in cand.iterdir() if d.is_dir() and (d / 'labels.npy').exists()]
            if len(subdirs) > 0:
                return cand
    raise FileNotFoundError("未能在项目中检索到包含 labels.npy 的 subject 数据文件夹。")

def evaluate_and_visualize_loso(pkl_path: str, data_root_dir: Path = None, target_idx: int = None, patient_id: str = None, record_id: str = None):
    if data_root_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        data_root_dir = find_data_root(project_root)

    logging.info(f"正在加载 LOSO 结果包: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        outputs = pickle.load(f)

    # 如果指定了 patient_id 或 record_id，按条件精准查找索引
    if patient_id is not None or record_id is not None:
        matched_indices = []
        for idx, item in enumerate(outputs):
            p_match = (patient_id is None) or (item.patient_id == patient_id)
            r_match = (record_id is None) or (item.record_id == record_id)
            if p_match and r_match:
                matched_indices.append(idx)
        
        if len(matched_indices) == 0:
            raise ValueError(f"未找到符合条件的结果: patient_id={patient_id}, record_id={record_id}")
        
        target_idx = matched_indices[0]
        logging.info(f"匹配到条件的序列索引为: {target_idx} (共找到 {len(matched_indices)} 条匹配片段)")
    elif target_idx is None:
        target_idx = 0

    if target_idx >= len(outputs):
        raise ValueError(f"索引超出范围，总共有 {len(outputs)} 条 Record 结果。")

    out = outputs[target_idx]
    patient = out.patient_id
    record = out.record_id
    times = out.timestamps_sec
    L = len(times)

    logging.info(f"正在分析受试者: {patient}, Record: {record}, 长度: {L}")

    subject_dir = data_root_dir / patient
    if not (subject_dir / 'labels.npy').exists():
        candidates = [d for d in data_root_dir.iterdir() if d.is_dir() and (d.name == patient or patient in d.name)]
        if len(candidates) > 0:
            subject_dir = candidates[0]
        else:
            raise FileNotFoundError(f"未能在 {data_root_dir} 下找到受试者文件夹: {patient}")

    true_labels = np.load(subject_dir / 'labels.npy', mmap_mode='r')
    rec_ids = np.load(subject_dir / 'sequence_records.npy', allow_pickle=False)

    mask = (rec_ids == record)
    slice_labels = true_labels[mask][:L]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f'Strictly Causal LOSO Inference | Subject: {patient} | Record: {record}', fontsize=14, y=0.98)

    # 1. HSMM 连续概率
    ax1 = axes[0]
    ax1.plot(times, out.hsmm_state_probabilities[:, 0], label='Interictal (0)', color='green', alpha=0.85, linewidth=1.8)
    ax1.plot(times, out.hsmm_state_probabilities[:, 1], label='Transition (1)', color='orange', alpha=0.85, linewidth=1.8)
    ax1.plot(times, out.hsmm_state_probabilities[:, 2], label='Preictal (2)', color='red', alpha=0.85, linewidth=1.8)
    ax1.set_ylabel('HSMM Probabilities')
    ax1.set_title('Continuous HSMM State Posterior', fontsize=11, pad=6)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left', frameon=True)

    # 2. 预测路径 vs TCN EndPoint Label (加亮显示 TCN 预警窗口)
    ax2 = axes[1]
    
    # 对 TCN Label == 1 的区域加亮背景，展示视觉对比
    is_positive = (slice_labels == 1)
    if np.any(is_positive):
        diff = np.diff(np.pad(is_positive.astype(int), (1, 1), 'constant'))
        p_starts = np.where(diff == 1)[0]
        p_ends = np.where(diff == -1)[0]
        for st_idx, ed_idx in zip(p_starts, p_ends):
            ed_idx = min(ed_idx, L - 1)
            ax2.axvspan(times[st_idx], times[ed_idx], color='gold', alpha=0.18, label='TCN Positive Window' if st_idx == p_starts[0] else "")

    ax2.plot(times, slice_labels, label='TCN GroundTruth Label (0:Inter, 1:Trans/Pre)', color='black', linestyle='--', linewidth=2.0)
    ax2.plot(times, out.hsmm_state_path, label='Inferred HSMM Path (0:Inter, 1:Trans, 2:Pre)', color='blue', linewidth=1.8, alpha=0.85)
    ax2.set_ylabel('State Class')
    ax2.set_title('Inferred State Path vs TCN Endpoint Label', fontsize=11, pad=6)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['0: Interictal', '1: Transition', '2: Preictal'])
    ax2.set_ylim(-0.2, 2.2)
    ax2.legend(loc='upper left', frameon=True)

    # 3. BOCPD 变化点概率
    ax3 = axes[2]
    ax3.plot(times, out.bocpd_change_probability, label='P(Change Point)', color='purple', linewidth=1.5)
    ax3.set_ylabel('Change Prob')
    ax3.set_xlabel('Endpoint Time (Seconds)')
    ax3.set_title('BOCPD Online Change Detection (Reset at Record Boundary)', fontsize=11, pad=6)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper left', frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    save_name = f'loso_vis_{patient}_{record}.png'
    plt.savefig(save_name, dpi=300)
    logging.info(f"可视化图表已保存至 {save_name}")
    plt.show()

def list_available_records(pkl_path: str):
    """打印 PKL 文件中包含的所有受试者和 Record 索引清单"""
    with open(pkl_path, 'rb') as f:
        outputs = pickle.load(f)
    print(f"\n=== {pkl_path} 中共包含 {len(outputs)} 条数据片段 ===")
    print(f"{'Index':<8} | {'Patient ID':<15} | {'Record ID':<15} | {'Timestamps Count':<10}")
    print("-" * 55)
    for idx, out in enumerate(outputs[:30]): # 打印前30条作为示例
        print(f"{idx:<8} | {out.patient_id:<15} | {out.record_id:<15} | {len(out.timestamps_sec):<10}")
    if len(outputs) > 30:
        print(f"... 还有 {len(outputs) - 30} 条记录未列出。")

if __name__ == "__main__":
    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parents[1]
    
    try:
        data_root = find_data_root(project_root)
        pkl_file = 'hbsi_outputs_loso_all.pkl'
        
        # 1. 打印前 30 条可用纪录的索引
        # list_available_records(pkl_file)
        
        # 2. 查看目标片段结果
        evaluate_and_visualize_loso(pkl_file, data_root_dir=data_root, target_idx=105)

    except Exception as e:
        print(f"可视化失败: {e}")