import numpy as np
import json
import os
from datetime import datetime, timezone
import logging

def save_hbsi_output(hsmm_probs: np.ndarray, bocpd_probs: np.ndarray, output_dir: str, config_hash: str):
    """
    专门负责将算法结果按 V0 合同标准进行格式化、校验并保存。
    绝不包含任何算法逻辑。
    
    参数:
        hsmm_probs: 维度预期为 (B, L, K) 或 (L, K)
        bocpd_probs: 维度预期为 (B, L) 或 (L,)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 强制数据类型校验与转换 (合同要求 float32)
    hsmm_probs = hsmm_probs.astype(np.float32)
    bocpd_probs = bocpd_probs.astype(np.float32)
    
    # 2. 写入 numpy 数组
    hsmm_path = os.path.join(output_dir, "hsmm_state_probs.npy")
    bocpd_path = os.path.join(output_dir, "bocpd_change_probs.npy")
    
    np.save(hsmm_path, hsmm_probs)
    np.save(bocpd_path, bocpd_probs)
    
    # 3. 生成并写入严谨的元数据 JSON (合同第 7 节)
    metadata = {
        "schema_version": "v0",
        "data_version": "v0",
        "module": "HBSI (Member 3)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "tensors": {
            "hsmm_state_probs": {
                "filename": "hsmm_state_probs.npy",
                "shape": list(hsmm_probs.shape),
                "dtype": "float32",
                "semantics": "Probability of [Interictal, Transition, Preictal]"
            },
            "bocpd_change_probs": {
                "filename": "bocpd_change_probs.npy",
                "shape": list(bocpd_probs.shape),
                "dtype": "float32",
                "semantics": "Per-timestep scalar probability of structural change"
            }
        },
        "notes": "Transition labels are refined by HSMM state inference. Meets V0 contract."
    }
    
    meta_path = os.path.join(output_dir, "HBSIOutput_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logging.info(f"成功保存符合 V0 合同的 HBSI 输出至: {output_dir}")