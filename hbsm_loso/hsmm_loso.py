import numpy as np
from scipy.special import logsumexp

class HSMM:
    """
    隐半马尔可夫模型 (极端临床惯性 Schmitt Trigger 迟滞版)
    完全基于对数空间前向滤波解决时序约束问题。
    使用 10 个隐状态实现：安全 -> 延时缓冲 -> 锁定预警 的不可逆病理演进。
    """
    def __init__(self, n_states=3, d_max=None):
        self.n_states = n_states
        # 10 个内部微观状态:
        # 0: Interictal (安全期)
        # 1~8: Transition (强制缓冲链，8 步约 20 秒)
        # 9: Preictal (发作前期)
        self.n_internal = 10  
        self.A = np.zeros((self.n_internal, self.n_internal))
        
        # ----------------------------------------------------
        # 极高惯性拓扑 (Extreme Inertia Topology)
        # ----------------------------------------------------
        # 1. 0态(安全期)：极其稳定。TCN 必须连续报高警 7秒以上 才能突破 1e-20 的惩罚。
        self.A[0, 0] = 1.0 - 1e-20
        self.A[0, 1] = 1e-20
        
        # 2. 1~8态(强制缓冲链)：进入后必须走完，耗时约 20秒 (取决于您的实际步长)
        for i in range(1, 9):
            self.A[i, i+1] = 1.0
            
        # 3. 9态(预警期)：极其黏滞。TCN 即使漏报掉回 0，HSMM 也会惯性维持警报 15秒以上！
        self.A[9, 9] = 1.0 - 1e-50
        self.A[9, 0] = 1e-50

    def fit_emissions(self, X, y=None):
        pass

    def _compute_emission_log_probs(self, risk_scores):
        """尖锐高斯惩罚映射，配合拓扑实现迟滞效应"""
        r = np.clip(risk_scores, 0.0, 1.0)
        
        # 状态0 期待 risk 为 0；激活态期待 risk 为 1
        # 放大幅度（15.0）用来平衡转移矩阵的巨大惩罚项
        LL_0 = -15.0 * (r - 0.0)**2
        LL_active = -15.0 * (r - 1.0)**2
        
        T = len(risk_scores)
        E_log = np.zeros((T, self.n_internal))
        
        E_log[:, 0] = LL_0
        for i in range(1, 10):
            E_log[:, i] = LL_active
            
        return E_log

    def forward_inference(self, X=None, risk_scores=None):
        """对数空间前向滤波 (严格时间序列推断)"""
        T = len(risk_scores)
        if T == 0: return np.zeros((0, self.n_states))
            
        E_log = self._compute_emission_log_probs(risk_scores)
        A_log = np.log(np.maximum(self.A, 1e-300))
        
        alpha = np.full((T, self.n_internal), -np.inf)
        alpha[0, 0] = E_log[0, 0] # 强制安全起手
        
        for t in range(1, T):
            for j in range(self.n_internal):
                alpha[t, j] = logsumexp(alpha[t-1, :] + A_log[:, j]) + E_log[t, j]
                
        alpha_norm = alpha - logsumexp(alpha, axis=1, keepdims=True)
        probs_internal = np.exp(alpha_norm)
        
        state_probs = np.zeros((T, 3))
        state_probs[:, 0] = probs_internal[:, 0]
        state_probs[:, 1] = np.sum(probs_internal[:, 1:9], axis=1) 
        state_probs[:, 2] = probs_internal[:, 9]
        
        return state_probs