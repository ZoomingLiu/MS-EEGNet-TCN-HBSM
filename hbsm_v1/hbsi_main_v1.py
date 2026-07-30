import numpy as np
import logging
from hsmm_v1 import HSMM
from bocpd_v1 import BOCPD
from hbsi_interface_v1 import package_hbsi_results, save_to_pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_tcn_npz_to_pkl(input_npz_path: str, output_pkl_path: str, is_training=False, trained_hsmm=None):
    """
    HBSI 主管线：读取组员2的 .npz，执行截断与推断，调用组员4的接口打包导出。
    """
    logging.info(f"=== 开始处理数据包: {input_npz_path} ===")
    
    # 1. 解包数据 (需与组员2确认具体 Keys，这里假设常规命名)
    data = np.load(input_npz_path, allow_pickle=True)
    tcn_embeds = data['embeddings']   # (B, L, H)
    tcn_risk = data['risk_scores']    # (B, L)
    labels = data['labels']           # (B, L)
    timestamps = data['timestamps']   # (B, L)
    
    B, L, H = tcn_embeds.shape
    
    # 构建 metadata
    metadata_list = []
    for b in range(B):
        split_type = "TRAIN" if is_training else "TEST" # 可根据实际传参调整
        metadata_list.append({
            'sequence_id': f"seq_{b:04d}",
            'split': split_type,
            'timestamps_sec': timestamps[b]
        })
        
    all_hsmm_probs = []
    all_bocpd_probs = []
    original_L_list = [L] * B

    # 2. 训练模式：拟合 HSMM
    if is_training:
        logging.info("训练模式：基于干净数据（排除 Ictal）拟合 HSMM ...")
        trained_hsmm = HSMM(n_states=3, d_max=50)
        clean_embeds = []
        clean_labels = []
        for b in range(B):
            ictal_idx = np.where(labels[b] == 3)[0]
            cutoff = ictal_idx[0] if len(ictal_idx) > 0 else L
            clean_embeds.append(tcn_embeds[b][:cutoff])
            clean_labels.append(labels[b][:cutoff])
        
        flat_embeds = np.vstack(clean_embeds)
        flat_labels = np.concatenate(clean_labels)
        trained_hsmm.fit_emissions(flat_embeds, flat_labels)
        
    if trained_hsmm is None:
        raise ValueError("推理模式必须提供 trained_hsmm 实例！")

    # 3. 逐序列推断
    logging.info("执行 HSMM 推断与 BOCPD 变化点检测 ...")
    for b in range(B):
        embeds_seq = tcn_embeds[b]
        risk_seq = tcn_risk[b]
        label_seq = labels[b]
        
        # 紧急避险逻辑：遇到 Ictal (3) 直接截断！
        ictal_idx = np.where(label_seq == 3)[0]
        cutoff = ictal_idx[0] if len(ictal_idx) > 0 else L
        
        valid_embeds = embeds_seq[:cutoff]
        valid_risk = risk_seq[:cutoff]
        
        # HSMM 推断
        hsmm_probs = trained_hsmm.forward_inference(valid_embeds)
        all_hsmm_probs.append(hsmm_probs)
        
        # BOCPD 在线推断
        bocpd = BOCPD(hazard_rate=0.01) # 每次新序列重置先验状态
        bocpd_probs = np.array([bocpd.update(x) for x in valid_risk])
        all_bocpd_probs.append(bocpd_probs)

    # 4. 打包并交接给组员4
    logging.info("开始调用接口规范打包数据 ...")
    hbsi_outputs = package_hbsi_results(metadata_list, all_hsmm_probs, all_bocpd_probs, original_L_list)
    save_to_pickle(hbsi_outputs, output_pkl_path)
    
    logging.info(f"=== 处理完成！请将 {output_pkl_path} 发送给组员 4 ===")
    return trained_hsmm

if __name__ == "__main__":
    # 本地 Dummy 测试示例 (确保您自己造假数据能跑通)
    # trained_model = process_tcn_npz_to_pkl('tcn_sequence_outputs_train.npz', 'hbsi_outputs_train.pkl', is_training=True)
    # process_tcn_npz_to_pkl('tcn_sequence_outputs_test.npz', 'hbsi_outputs_test.pkl', is_training=False, trained_hsmm=trained_model)
    pass