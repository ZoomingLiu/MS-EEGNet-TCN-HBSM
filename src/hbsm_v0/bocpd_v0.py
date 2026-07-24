import numpy as np
from scipy.stats import t
import logging

class BOCPD:
    """
    贝叶斯在线变化点检测核心类 (V0 工程版)
    仅包含数学运算和状态更新，移除所有可视化和 IO 代码。
    """
    def __init__(self, hazard_rate=0.01, mean0=0.0, var0=1.0, alpha0=1.0, beta0=1.0):
        self.hazard_rate = hazard_rate
        self.mean0 = mean0
        self.var0 = var0      
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.reset_state()

    def reset_state(self):
        """重置内部状态，在检测到发作或序列间断时调用"""
        self.t = 0
        self.R = np.array([1.0]) 
        
        self.means = np.array([self.mean0])
        self.kappas = np.array([self.var0])
        self.alphas = np.array([self.alpha0])
        self.betas = np.array([self.beta0])
        
        self.change_probabilities = []

    def _student_t_pdf(self, x, mean, kappa, alpha, beta):
        """计算给定参数下 Student-t 的预测概率密度"""
        var = beta * (kappa + 1) / (alpha * kappa)
        df = 2 * alpha
        scale = np.sqrt(var)
        scale = np.maximum(scale, 1e-8)
        return t.pdf(x, df, loc=mean, scale=scale)

    def update(self, x):
        """
        输入 x: t时刻的观测值
        输出: P(Change_t)
        """
        self.t += 1
        
        pred_probs = self._student_t_pdf(x, self.means, self.kappas, self.alphas, self.betas)
        pred_probs = np.maximum(pred_probs, 1e-100) 
        
        growth_probs = self.R * pred_probs * (1 - self.hazard_rate)
        change_prob = np.sum(self.R * pred_probs * self.hazard_rate)
        
        self.R = np.append([change_prob], growth_probs)
        self.R /= np.sum(self.R)
        
        current_cp_prob = self.R[0]
        self.change_probabilities.append(current_cp_prob)
        
        new_means = (self.kappas * self.means + x) / (self.kappas + 1)
        new_kappas = self.kappas + 1
        new_alphas = self.alphas + 0.5
        new_betas = self.betas + (self.kappas * (x - self.means)**2) / (2 * (self.kappas + 1))
        
        self.means = np.append([self.mean0], new_means)
        self.kappas = np.append([self.var0], new_kappas)
        self.alphas = np.append([self.alpha0], new_alphas)
        self.betas = np.append([self.beta0], new_betas)
        
        return current_cp_prob