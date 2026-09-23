import numpy as np

class BRCD_Exact:
    """
    BRCD (Bayesian Root Cause Discovery - ICML 2026):
    Computes the posterior probability of candidate root causes using the
    Interventional Causal Mechanism Shift score:
      Score(R) = || E_int[R | Pa(R)] - E_obs[R | Pa(R)] ||^2 / Var_obs(R | Pa(R))
    """
    def __init__(self, causal_system):
        self.system = causal_system
        self.d = causal_system.d
        self.candidate_indices = causal_system.candidate_indices
        self.node_names = causal_system.node_names

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        m = len(D_int)
        
        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            
            if len(parents) == 0:
                mu_obs = np.mean(D_obs[:, r_idx])
                var_obs = max(1e-5, np.var(D_obs[:, r_idx]))
                mu_int = np.mean(D_int[:, r_idx])
                
                # Standardized mean shift squared (Wald test statistic / Bayesian log likelihood ratio)
                shift_score = ((mu_int - mu_obs) ** 2) / var_obs
            else:
                X_par_obs = np.hstack([np.ones((len(D_obs), 1)), D_obs[:, parents]])
                beta_obs = np.linalg.lstsq(X_par_obs, D_obs[:, r_idx], rcond=None)[0]
                var_obs = max(1e-5, np.var(D_obs[:, r_idx] - X_par_obs @ beta_obs))
                
                X_par_int = np.hstack([np.ones((m, 1)), D_int[:, parents]])
                pred_obs = X_par_int @ beta_obs
                res_int = D_int[:, r_idx] - pred_obs
                
                shift_score = (np.mean(res_int) ** 2) / var_obs

            scores[r_idx] = float(shift_score)

        # Softmax posterior
        max_s = max(scores.values())
        unnorm = {r: np.exp(np.clip(s - max_s, -50, 50)) for r, s in scores.items()}
        total_p = sum(unnorm.values()) + 1e-18
        
        ranking = []
        for r_idx in self.candidate_indices:
            p_val = unnorm[r_idx] / total_p
            ranking.append((self.node_names[r_idx], p_val))
            
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
