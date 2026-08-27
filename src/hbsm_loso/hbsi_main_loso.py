import numpy as np
from pathlib import Path
import logging

from hsmm_loso import HSMM
from bocpd_loso import BOCPD
from hbsi_interface_loso import HBSIOutput, save_to_pickle

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
    raise FileNotFoundError("未能在项目中检索到包含 labels.npy 的 subject 数据文件夹，请检查路径。")

def load_subject_data(subject_dir: Path):
    risk = np.load(subject_dir / 'tcn_risk.npy', mmap_mode='r')
    labels = np.load(subject_dir / 'labels.npy', mmap_mode='r')
    records = np.load(subject_dir / 'sequence_records.npy', allow_pickle=False)
    endpoint_seconds = np.load(subject_dir / 'sequence_end_seconds.npy', mmap_mode='r')

    N = len(risk)
    return {
        'risk': risk,
        'labels': labels,
        'records': records,
        'endpoint_seconds': endpoint_seconds,
        'N': N
    }

def run_loso_pipeline(data_dir_path: Path, output_pkl_path: str):
    subject_dirs = sorted([d for d in data_dir_path.iterdir() if d.is_dir() and (d / 'labels.npy').exists()])
    logging.info(f"=== 开始极速 LOSO 推断管线 (Subject拼接版)，共发现 {len(subject_dirs)} 个受试者 ===")
    
    all_loso_outputs = []

    for test_dir in subject_dirs:
        test_subject_id = test_dir.name
        logging.info(f">>> 正在推断测试受试者: {test_subject_id} ...")

        hsmm_model = HSMM(n_states=3)

        test_data = load_subject_data(test_dir)
        risk = test_data['risk']
        records = test_data['records']
        endpoint_seconds = test_data['endpoint_seconds']

        # 按 record 切分连续片段的索引
        change_points = np.flatnonzero(records[1:] != records[:-1]) + 1
        record_slices = np.split(np.arange(len(records)), change_points)

        # 用于存储该受试者所有切片的推断结果
        subj_hsmm_probs = []
        subj_hsmm_path = []
        subj_bocpd_probs = []
        subj_times = []

        for slice_idx, indices in enumerate(record_slices):
            L = len(indices)
            if L == 0: continue

            p_slice = risk[indices]
            t_slice = endpoint_seconds[indices]

            # 1. HSMM 推断 (每段 Record 都会用全新的 0 态起手)
            hsmm_probs = hsmm_model.forward_inference(X=None, risk_scores=p_slice)
            hsmm_probs = np.clip(hsmm_probs, 1e-10, 1.0)
            hsmm_probs = hsmm_probs / hsmm_probs.sum(axis=1, keepdims=True)
            hsmm_path = np.argmax(hsmm_probs, axis=1)

            # 2. BOCPD 推断 (每段 Record 都会重置方差和状态先验)
            bocpd = BOCPD()
            bocpd_probs = np.array([bocpd.update(x) for x in p_slice], dtype=np.float32)

            subj_hsmm_probs.append(hsmm_probs)
            subj_hsmm_path.append(hsmm_path)
            subj_bocpd_probs.append(bocpd_probs)
            subj_times.append(t_slice)

        # =========================================================
        # 核心修改：在受试者级别进行无缝拼接！迎合组员4的 Shape 需求
        # =========================================================
        final_hsmm_probs = np.concatenate(subj_hsmm_probs, axis=0)
        final_hsmm_path = np.concatenate(subj_hsmm_path, axis=0)
        final_bocpd_probs = np.concatenate(subj_bocpd_probs, axis=0)
        final_times = np.concatenate(subj_times, axis=0)

        # 强校验：拼接后的总长度必须等于 TCN 输入的总长度
        assert len(final_hsmm_probs) == test_data['N'], f"拼接长度错误: {len(final_hsmm_probs)} != {test_data['N']}"

        output_obj = HBSIOutput(
            sequence_id=f"{test_subject_id}_merged",
            patient_id=str(test_subject_id),
            record_id="ALL_RECORDS_MERGED",  # 告知组员4已合并
            split="LOSO_HELD_OUT",
            timestamps_sec=final_times.astype(np.float64),
            hsmm_state_probabilities=final_hsmm_probs.astype(np.float32),
            hsmm_state_path=final_hsmm_path.astype(np.int64),
            bocpd_change_probability=final_bocpd_probs,
            valid_mask=np.ones(test_data['N'], dtype=bool)
        )
        all_loso_outputs.append(output_obj)
        logging.info(f"    完成拼接！总长度: {test_data['N']} 步。")

    save_to_pickle(all_loso_outputs, output_pkl_path)
    logging.info(f"🎉 极速推断与拼接完成！已导出 {len(all_loso_outputs)} 个受试者大对象至 {output_pkl_path}")

if __name__ == "__main__":
    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parents[1]
    
    data_root_dir = find_data_root(project_root)
    run_loso_pipeline(data_root_dir, 'hbsi_outputs_loso_subject_merged.pkl')