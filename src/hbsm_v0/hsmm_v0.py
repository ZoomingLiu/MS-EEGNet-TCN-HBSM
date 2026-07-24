import numpy as np
from scipy.special import logsumexp
from scipy.stats import poisson, multivariate_normal
import logging

class HSMM:
    """
    隐半马尔可夫模型核心类 (V0 工程版)
    仅包含状态推断与参数计算。
    """
    def __init__(self, n_states=3, d_max=50):
        self.n_states = n_states
        self.d_max = d_max
        
        self.transition_matrix = np.array([
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0], 
            [0.0, 0.0, 0.0]  
        ])
        self.initial_probs = np.array([0.98, 0.01, 0.01])
        self.duration_lambdas = np.array([30.0, 15.0, 20.0]) 
        
        self.emission_means = None
        self.emission_covs = None

    def _get_duration_log_probs(self):
        D_log = np.zeros((self.n_states, self.d_max))
        for k in range(self.n_states):
            d_vals = np.arange(1, self.d_max + 1)
            probs = poisson.pmf(d_vals, self.duration_lambdas[k])
            D_log[k, :] = np.log(np.maximum(probs, 1e-300))
        return D_log

    def _compute_emission_log_probs(self, X):
        T = len(X)
        E_log = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            rv = multivariate_normal(self.emission_means[k], self.emission_covs[k])
            E_log[:, k] = rv.logpdf(X)
        return E_log

    def fit_emissions_synthetic(self, X, y):
        """
        TODO: 未来在此实现 EM 算法拟合。
        目前暂时使用传入的弱标签进行均值和协方差的高斯拟合。
        """
        _, dim = X.shape
        self.emission_means = np.zeros((self.n_states, dim))
        self.emission_covs = np.zeros((self.n_states, dim, dim))
        
        for k in range(self.n_states):
            X_k = X[y == k]
            if len(X_k) > 0:
                self.emission_means[k] = np.mean(X_k, axis=0)
                self.emission_covs[k] = np.cov(X_k, rowvar=False) + np.eye(dim) * 1e-6
            else:
                self.emission_means[k] = np.zeros(dim)
                self.emission_covs[k] = np.eye(dim)

    def forward_inference(self, X):
        """执行显式持续时间的 HSMM 前向推断"""
        if self.emission_means is None:
            raise ValueError("请先调用 fit 拟合发射概率参数。")
            
        T = len(X)
        D_log = self._get_duration_log_probs()
        E_log = self._compute_emission_log_probs(X)
        A_log = np.log(np.maximum(self.transition_matrix, 1e-300))
        pi_log = np.log(np.maximum(self.initial_probs, 1e-300))
        
        alpha_log = np.full((T, self.n_states), -np.inf)
        E_log_cumsum = np.vstack([np.zeros((1, self.n_states)), np.cumsum(E_log, axis=0)])
        
        for t in range(T):
            for j in range(self.n_states):
                for d in range(1, min(t + 1, self.d_max) + 1):
                    obs_log_prob = E_log_cumsum[t+1, j] - E_log_cumsum[t+1-d, j]
                    dur_log_prob = D_log[j, d-1]
                    
                    if t - d < 0: 
                        term = pi_log[j] + dur_log_prob + obs_log_prob
                        alpha_log[t, j] = np.logaddexp(alpha_log[t, j], term)
                    else: 
                        for i in range(self.n_states):
                            if A_log[i, j] > -np.inf:
                                term = alpha_log[t-d, i] + A_log[i, j] + dur_log_prob + obs_log_prob
                                alpha_log[t, j] = np.logaddexp(alpha_log[t, j], term)

        state_probs = np.exp(alpha_log - logsumexp(alpha_log, axis=1, keepdims=True))
        state_probs[np.isnan(state_probs)] = 1.0 / self.n_states
        
        return state_probs