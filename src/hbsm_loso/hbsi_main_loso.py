import numpy as np
from pathlib import Path
import logging
import argparse

try:  # Support both ``python -m hbsm_loso.hbsi_main_loso`` and direct execution.
    from .hsmm_loso import HSMM
    from .bocpd_loso import BOCPD
    from .hbsi_interface_loso import HBSIOutput, save_to_pickle
    from .risk_calibration_loso import CausalRiskCalibrator
except ImportError:  # pragma: no cover - direct-script convenience
    from hsmm_loso import HSMM
    from bocpd_loso import BOCPD
    from hbsi_interface_loso import HBSIOutput, save_to_pickle
    from risk_calibration_loso import CausalRiskCalibrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WINDOW_STEP_SECONDS = 2.5
TRANSITION_STEPS = 8       # 20 seconds of sustained high evidence
RELEASE_STEPS = 6          # 15 seconds of sustained low evidence
EXPECTED_CHANGE_SECONDS = 3600.0

def find_data_root(start_dir: Path) -> Path:
    data_dir = start_dir / 'data'
    if data_dir.exists() and data_dir.is_dir():
        return data_dir
    raise FileNotFoundError(f"未能在项目根目录下找到 data 文件夹: {data_dir}")

def load_subject_data(subject_dir: Path):
    risk = np.load(subject_dir / 'tcn_risk.npy', mmap_mode='r')
    embeddings = np.load(subject_dir / 'tcn_embedding_128.npy', mmap_mode='r')
    logits = np.load(subject_dir / 'tcn_logits.npy', mmap_mode='r')
    labels = np.load(subject_dir / 'labels.npy', mmap_mode='r')
    records = np.load(subject_dir / 'sequence_records.npy', allow_pickle=False)
    endpoint_seconds = np.load(subject_dir / 'sequence_end_seconds.npy', mmap_mode='r')

    N = len(risk)
    if not (len(labels) == len(records) == len(endpoint_seconds) == len(embeddings) == len(logits) == N):
        raise ValueError(f"{subject_dir}: input arrays must have identical lengths.")
    if embeddings.shape != (N, 128) or logits.shape != (N, 2):
        raise ValueError(f"{subject_dir}: expected embeddings (N, 128) and logits (N, 2).")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError(f"{subject_dir}: tcn_risk must be finite and lie in [0, 1].")
    return {
        'risk': risk,
        'labels': labels,
        'records': records,
        'endpoint_seconds': endpoint_seconds,
        'N': N
    }

def run_loso_pipeline(data_dir_path: Path, output_pkl_path: str | Path):
    subject_dirs = sorted([d for d in data_dir_path.iterdir() if d.is_dir() and (d / 'labels.npy').exists()])
    logging.info(f"=== 开始极速 LOSO 推断管线 (Subject拼接版)，共发现 {len(subject_dirs)} 个受试者 ===")
    
    all_loso_outputs = []

    for test_dir in subject_dirs:
        test_subject_id = test_dir.name
        logging.info(f">>> 正在推断测试受试者: {test_subject_id} ...")

        test_data = load_subject_data(test_dir)
        risk = test_data['risk']
        records = test_data['records']
        endpoint_seconds = test_data['endpoint_seconds']

        # 按 record 切分连续片段的索引
        change_points = np.flatnonzero(records[1:] != records[:-1]) + 1
        record_slices = np.split(np.arange(len(records)), change_points)

        if test_data['N'] == 0:
            logging.warning("    跳过空受试者: %s", test_subject_id)
            continue

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

            # 1. Fold 间 TCN 概率偏移很大，因此先做仅依赖过去值的
            # record-local 基线校正；不读取 held-out 标签或未来数据。
            calibrator = CausalRiskCalibrator(step_seconds=WINDOW_STEP_SECONDS)
            calibrated_evidence, logit_risk = calibrator.transform(p_slice)

            # 2. 每个物理 record 使用独立 HSMM 实例，显式隔离持续时间状态。
            # 在 2.5 秒步长下，8 个过渡步约对应 20 秒连续高证据，
            # 6 个释放步约对应 15 秒连续低证据。
            hsmm_model = HSMM(
                n_states=3,
                transition_steps=TRANSITION_STEPS,
                release_steps=RELEASE_STEPS,
            )
            hsmm_probs = hsmm_model.forward_inference(
                X=None, risk_scores=calibrated_evidence
            )
            hsmm_probs = np.clip(hsmm_probs, 1e-10, 1.0)
            hsmm_probs = hsmm_probs / hsmm_probs.sum(axis=1, keepdims=True)
            hsmm_path = np.argmax(hsmm_probs, axis=1)

            # 3. BOCPD 在 logit-risk 上检测统计跃迁，并在 record 边界重置。
            bocpd = BOCPD(
                hazard_rate=WINDOW_STEP_SECONDS / EXPECTED_CHANGE_SECONDS,
                mean0=float(logit_risk[0]),
                var0=0.1,
                alpha0=2.0,
                beta0=1.0,
                max_run_length=512,
                change_window=3,
            )
            bocpd_probs = np.array(
                [bocpd.update(value) for value in logit_risk], dtype=np.float32
            )

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
        if len(final_hsmm_probs) != test_data['N']:
            raise RuntimeError(f"拼接长度错误: {len(final_hsmm_probs)} != {test_data['N']}")

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

def main(argv=None):
    """CLI wrapper; numerical inference and the HBSIOutput contract are unchanged."""
    parser = argparse.ArgumentParser(description="Run the legacy LOSO HBSI risk filter.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing subject_* deliveries.")
    parser.add_argument("--output", type=Path, help="Defaults to outputs/hbsi/hbsi_loso_output_final.pkl.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing result.")
    args = parser.parse_args(argv)
    if args.data_dir is None or args.output is None:
        try:
            from eeg_pipeline.paths import hbsi_data_root, project_root
            root = project_root()
            data_dir = args.data_dir if args.data_dir is not None else hbsi_data_root(root)
        except ImportError:  # Direct script without editable installation.
            root = Path(__file__).resolve().parents[2]
            data_dir = args.data_dir if args.data_dir is not None else find_data_root(root)
        output = args.output if args.output is not None else root / "outputs" / "hbsi" / "hbsi_loso_output_final.pkl"
    else:
        data_dir, output = args.data_dir, args.output
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output}. Choose another --output or use --overwrite.")
    if not data_dir.is_dir():
        parser.error(f"Data directory does not exist: {data_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_loso_pipeline(data_dir, output)


if __name__ == "__main__":
    main()
