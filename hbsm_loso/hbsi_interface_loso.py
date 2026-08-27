import pickle
import numpy as np
import logging
from dataclasses import dataclass
from typing import List

@dataclass
class HBSIOutput:
    sequence_id: str
    patient_id: str
    record_id: str
    split: str  
    timestamps_sec: np.ndarray
    hsmm_state_probabilities: np.ndarray
    hsmm_state_path: np.ndarray
    bocpd_change_probability: np.ndarray
    valid_mask: np.ndarray

def package_hbsi_results(metadata_list: list, all_hsmm_probs: list, all_bocpd_probs: list, original_L_list: list) -> List[HBSIOutput]:
    """
    将结果打包为组员4要求的结构。针对遇到 Ictal 截断的序列，使用零值或保持状态进行长度补齐，
    以确保最终输出的维度 (L,) 严格对应 TCN 的原始时间戳长度。
    """
    hbsi_outputs = []
    
    for i, meta in enumerate(metadata_list):
        orig_L = original_L_list[i]
        hsmm_probs = all_hsmm_probs[i] 
        bocpd_probs = all_bocpd_probs[i] 
        
        cut_L = len(hsmm_probs)
        
        # 1. 补全因 Ictal 被截断的序列尾部
        final_hsmm = np.zeros((orig_L, 3), dtype=np.float32)
        final_bocpd = np.zeros(orig_L, dtype=np.float32)
        
        if cut_L > 0:
            final_hsmm[:cut_L, :] = hsmm_probs
            final_bocpd[:cut_L] = bocpd_probs
            # Ictal 部分不输出发作前状态概率，可设为安全基线或 0
            if cut_L < orig_L:
                final_hsmm[cut_L:, 0] = 1.0 
        else:
            final_hsmm[:, 0] = 1.0 # 极端情况防崩溃
            
        # 2. 强校验：归一化与裁剪
        final_hsmm = np.clip(final_hsmm, 1e-10, 1.0)
        final_hsmm = final_hsmm / final_hsmm.sum(axis=1, keepdims=True)
        final_bocpd = np.clip(final_bocpd, 0.0, 1.0)
        
        # 3. 提取最可能状态路径
        hsmm_path = np.argmax(final_hsmm, axis=1)
        
        # 4. 创建 Dataclass
        output_obj = HBSIOutput(
            sequence_id=meta.get('sequence_id', f"seq_{i}"),
            patient_id=meta.get('patient_id', 'unknown'),
            record_id=meta.get('record_id', 'unknown'),
            split=meta.get('split', 'UNKNOWN'),
            timestamps_sec=meta['timestamps_sec'],
            hsmm_state_probabilities=final_hsmm.astype(np.float32),
            hsmm_state_path=hsmm_path.astype(np.int64),
            bocpd_change_probability=final_bocpd.astype(np.float32),
            valid_mask=np.ones(orig_L, dtype=bool)
        )
        hbsi_outputs.append(output_obj)
        
    return hbsi_outputs

def save_to_pickle(hbsi_outputs: List[HBSIOutput], filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(hbsi_outputs, f)
    logging.info(f"成功导出 {len(hbsi_outputs)} 条序列的 HBSIOutput 列表至: {filepath}")