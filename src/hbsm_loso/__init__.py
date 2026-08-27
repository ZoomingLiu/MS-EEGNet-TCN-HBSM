import pickle
import numpy as np
import logging
from dataclasses import dataclass
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================
# 1. 严格按照成员 4 要求的结构定义 DataClass
# =====================================================================
@dataclass
class HBSIOutput:
    sequence_id: str
    patient_id: str
    record_id: str
    split: str  # 'TRAIN', 'VAL', or 'TEST'
    timestamps_sec: np.ndarray
    hsmm_state_probabilities: np.ndarray
    hsmm_state_path: np.ndarray
    bocpd_change_probability: np.ndarray
    valid_mask: np.ndarray

# =====================================================================
# 2. 核心打包与校验逻辑
# =====================================================================
def package_hbsi_results(tcn_metadata_list: list, all_hsmm_probs: list, all_bocpd_probs: list) -> List[HBSIOutput]:
    """
    将 HBSI 的计算结果与成员 2 的元数据合并，进行安全校验后打包为 HBSIOutput 列表。
    
    参数:
        tcn_metadata_list: 成员 2 传来的序列元数据字典列表 (包含 patient_id, timestamps_sec 等)
        all_hsmm_probs: 对应的 HSMM 概率数组列表
        all_bocpd_probs: 对应的 BOCPD 概率数组列表
    """
    hbsi_outputs = []
    
    for i, meta in enumerate(tcn_metadata_list):
        L = len(meta['timestamps_sec'])
        
        # 1. 提取当前序列的概率
        hsmm_probs = all_hsmm_probs[i] # 预期 shape (L, 3)
        bocpd_probs = all_bocpd_probs[i] # 预期 shape (L,)
        
        # 2. 安全校验与强制约束 (满足成员 4 的额外要求)
        # 约束 A: hsmm_state_probabilities 每一行和必须为 1
        # 防止微小的浮点数误差，强行归一化
        hsmm_probs = np.clip(hsmm_probs, 1e-10, 1.0)
        hsmm_probs = hsmm_probs / hsmm_probs.sum(axis=1, keepdims=True)
        
        # 约束 B: hsmm_state_path 必须是 (L,) 的最可能状态序列
        hsmm_path = np.argmax(hsmm_probs, axis=1)
        
        # 约束 C: bocpd_change_probability 必须在 [0, 1] 内
        bocpd_probs = np.clip(bocpd_probs, 0.0, 1.0)
        
        # 3. 创建输出对象
        output_obj = HBSIOutput(
            sequence_id=meta['sequence_id'],
            patient_id=meta['patient_id'],
            record_id=meta['record_id'],
            split=meta['split'],
            timestamps_sec=meta['timestamps_sec'], # 直接透传成员 2 的时间戳，保证 100% 对齐
            hsmm_state_probabilities=hsmm_probs.astype(np.float32),
            hsmm_state_path=hsmm_path.astype(np.int64),
            bocpd_change_probability=bocpd_probs.astype(np.float32),
            valid_mask=meta.get('valid_mask', np.ones(L, dtype=bool)) # 如果成员2没给，默认全为True
        )
        
        hbsi_outputs.append(output_obj)
        
    return hbsi_outputs

# =====================================================================
# 3. 保存至 pkl 文件
# =====================================================================
def save_to_pickle(hbsi_outputs: List[HBSIOutput], filepath: str = "hbsi_outputs.pkl"):
    """将对象列表保存为 pickle 文件供成员 4 读取"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(hbsi_outputs, f)
        logging.info(f"成功将 {len(hbsi_outputs)} 条序列数据打包保存至: {filepath}")
    except Exception as e:
        logging.error(f"保存 pickle 文件失败: {e}")

# =====================================================================
# 4. 模拟使用示例 (如何嵌入到您的主流程中)
# =====================================================================
if __name__ == "__main__":
    # 假设这是从成员 2 传来的数据中提取的元数据
    mock_metadata = [
        {
            "sequence_id": "seq_001",
            "patient_id": "chb01",
            "record_id": "chb01_03",
            "split": "TEST",
            "timestamps_sec": np.arange(0, 50, 5) # 10 个 5秒 窗口的时间戳
        }
    ]
    
    # 假设这是您的模型刚刚跑出的结果
    mock_hsmm = [np.random.dirichlet(np.ones(3), size=10)] # shape (10, 3)
    mock_bocpd = [np.random.uniform(0, 0.05, size=10)]     # shape (10,)
    
    # 执行打包
    final_outputs = package_hbsi_results(mock_metadata, mock_hsmm, mock_bocpd)
    
    # 导出文件
    save_to_pickle(final_outputs, "hbsi_outputs.pkl")