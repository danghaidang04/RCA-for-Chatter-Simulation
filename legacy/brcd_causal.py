import numpy as np
import scipy.stats as stats
import networkx as nx

class BRCD_Causal:
    """
    Complete BRCD Engine with Causal Mechanism Invariance:
    Scores candidate R based on direct mechanism shift and downstream causal consistency.
    """
    def __init__(self, causal_system):
        self.system = causal_system
        self.d = causal_system.d
        self.candidate_indices = causal_system.candidate_indices
        self.node_names = causal_system.node_names

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        m = len(D_int)
        
        obs_models = {}
        for i in range(self.d):
            parents = list(dag.predecessors(i))
            if len(parents) == 0:
                mu = np.mean(D_obs[:, i])
                var = max(1e-4, np.var(D_obs[:, i]))
                obs_models[i] = {"type": "root", "mu": mu, "var": var}
            else:
                X_par = np.hstack([np.ones((len(D_obs), 1)), D_obs[:, parents]])
                beta = np.linalg.lstsq(X_par, D_obs[:, i], rcond=None)[0]
                residuals = D_obs[:, i] - X_par @ beta
                var = max(1e-4, np.var(residuals))
                obs_models[i] = {"type": "child", "beta": beta, "var": var, "parents": parents}

        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            
            if len(parents) == 0:
                mu_obs = obs_models[r_idx]["mu"]
                var_obs = obs_models[r_idx]["var"]
                mu_int = np.mean(D_int[:, r_idx])
                direct_shift = ((mu_int - mu_obs) ** 2) / var_obs
            else:
                parents = obs_models[r_idx]["parents"]
                beta = obs_models[r_idx]["beta"]
                var_obs = obs_models[r_idx]["var"]
                X_par_int = np.hstack([np.ones((m, 1)), D_int[:, parents]])
                pred = X_par_int @ beta
                res = D_int[:, r_idx] - pred
                direct_shift = (np.mean(res) ** 2) / var_obs
                
            descendants = nx.descendants(dag, r_idx)
            causal_relevance = 1.0 + 0.1 * len(descendants)
            scores[r_idx] = float(direct_shift * causal_relevance)

        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
