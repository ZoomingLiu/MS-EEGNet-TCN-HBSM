import numpy as np
from scipy.stats import t

class BOCPD:
    """
    贝叶斯在线变化点检测 (标准 Student-t 解除底噪封印版)
    """
    def __init__(self, hazard_rate=0.01, mean0=0.5, var0=0.05, alpha0=2.0, beta0=0.05):
        self.hazard_rate = hazard_rate
        self.mean0 = mean0
        self.var0 = var0      
        self.alpha0 = alpha0
        self.beta0 = beta0 
        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.R = np.array([1.0]) 
        
        self.means = np.array([self.mean0])
        self.kappas = np.array([self.var0])
        self.alphas = np.array([self.alpha0])
        self.betas = np.array([self.beta0])

    def _student_t_pdf(self, x, mean, kappa, alpha, beta):
        var = beta * (kappa + 1) / (alpha * kappa)
        # 解除 0.1 的强制底噪封印！只保留极小的 1e-4 防止除零错误。
        # 让 BOCPD 恢复对结构性分布突变的敏锐嗅觉！
        var = np.maximum(var, 1e-4) 
        df = 2 * alpha
        scale = np.sqrt(var)
        return t.pdf(x, df, loc=mean, scale=scale)

    def update(self, x):
        self.t += 1
        
        pred_probs = self._student_t_pdf(x, self.means, self.kappas, self.alphas, self.betas)
        pred_probs = np.maximum(pred_probs, 1e-100) 
        
        growth_probs = self.R * pred_probs * (1 - self.hazard_rate)
        change_prob = np.sum(self.R * pred_probs * self.hazard_rate)
        
        self.R = np.append([change_prob], growth_probs)
        
        r_sum = np.sum(self.R)
        if r_sum > 0:
            self.R /= r_sum
        else:
            self.R = np.zeros_like(self.R)
            self.R[0] = 1.0
        
        # 聚合最近 3 步的突变概率，输出清晰的尖峰
        cp_prob = float(np.sum(self.R[:min(3, len(self.R))]))
        
        new_means = (self.kappas * self.means + x) / (self.kappas + 1)
        new_kappas = self.kappas + 1
        new_alphas = self.alphas + 0.5
        new_betas = self.betas + (self.kappas * (x - self.means)**2) / (2 * (self.kappas + 1))
        
        self.means = np.append([self.mean0], new_means)
        self.kappas = np.append([self.var0], new_kappas)
        self.alphas = np.append([self.alpha0], new_alphas)
        self.betas = np.append([self.beta0], new_betas)
        
        return cp_prob