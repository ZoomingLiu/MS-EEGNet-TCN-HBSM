import numpy as np
import logging
import os

# 导入拆分好的工程化模块
from hsmm_v0 import HSMM
from bocpd_v0 import BOCPD
from hbsi_interface_v0 import save_hbsi_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_batch(tcn_embeds_batch, tcn_risk_batch, dummy_labels_batch, output_dir, config_hash="test_hash_123"):
    """
    HBSI 主流水线调度器。
    接收来自成员2的批量输入，循环调用 HSMM 和 BOCPD，打包结果并输出。
    """
    B, L, H = tcn_embeds_batch.shape
    logging.info(f"开始处理 Batch 数据: BatchSize={B}, 序列长度={L}")
    
    all_hsmm_probs = []
    all_bocpd_probs = []
    
    # 对 Batch 中的每一条序列进行处理
    for b in range(B):
        embeds_seq = tcn_embeds_batch[b] # (L, H)
        risk_seq = tcn_risk_batch[b]     # (L,)
        labels_seq = dummy_labels_batch[b] # 临时方案，用于 fit
        
        # --- 1. 执行 BOCPD 算法 ---
        bocpd = BOCPD(hazard_rate=0.01)
        # 逐时间步推断并收集概率
        bocpd_probs = np.array([bocpd.update(x) for x in risk_seq])
        all_bocpd_probs.append(bocpd_probs)
        
        # --- 2. 执行 HSMM 算法 ---
        hsmm = HSMM(n_states=3, d_max=50)
        # TODO: 未来需接入真实的 EM 算法拟合。目前使用伪标签完成参数初始化。
        hsmm.fit_emissions_synthetic(embeds_seq, labels_seq)
        hsmm_probs = hsmm.forward_inference(embeds_seq)
        all_hsmm_probs.append(hsmm_probs)
        
    # 合并 Batch 结果
    final_hsmm_array = np.array(all_hsmm_probs)
    final_bocpd_array = np.array(all_bocpd_probs)
    
    # --- 3. 调用质检与打包接口 ---
    save_hbsi_output(
        hsmm_probs=final_hsmm_array, 
        bocpd_probs=final_bocpd_array, 
        output_dir=output_dir, 
        config_hash=config_hash
    )

if __name__ == "__main__":
    # 这个入口仅供您本地独立测试，或者用 dummy data 联调时使用
    logging.info("启动 HBSI 本地独立测试模式...")
    
    # 模拟成员 2 传入的假数据 (B=2, L=600, H=128)
    mock_embeds = np.random.randn(2, 600, 128).astype(np.float32)
    mock_risk = np.random.randn(2, 600).astype(np.float32)
    mock_labels = np.random.randint(0, 3, size=(2, 600))
    
    output_target = "./final_hbsi_results"
    
    process_batch(
        tcn_embeds_batch=mock_embeds,
        tcn_risk_batch=mock_risk,
        dummy_labels_batch=mock_labels,
        output_dir=output_target
    )
    
    logging.info(f"测试完成。请检查 {output_target} 目录下的文件。")