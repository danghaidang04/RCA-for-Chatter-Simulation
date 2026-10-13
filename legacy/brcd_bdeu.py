import numpy as np
import scipy.stats as stats
from scipy.special import gammaln

class BRCD_BDeu:
    """
    BRCD with BDeu Dirichlet-Multinomial Prequential Scoring (BRCD Appendix D & F):
    - Discretizes continuous metrics into K quantile states based on normal baseline.
    - Captures non-linear causal dynamics without linear parametric bias.
    - Evaluates exact marginal log-likelihood ratio for intervention target R:
        log P(D_int | G, R) = sum_{X in V} log BDeu(X | Pa(X), F)
    """
    def __init__(self, causal_system, n_bins=5, alpha_star=1.0):
        self.system = causal_system
        self.d = causal_system.d
        self.candidate_indices = causal_system.candidate_indices
        self.node_names = causal_system.node_names
        self.n_bins = n_bins
        self.alpha_star = alpha_star

    def _discretize(self, D_obs, D_int):
        bins = []
        D_obs_disc = np.zeros_like(D_obs, dtype=int)
        D_int_disc = np.zeros_like(D_int, dtype=int)
        
        for j in range(self.d):
            # Fit quantile bins on normal data
            quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            b = np.percentile(D_obs[:, j], quantiles)
            b = np.unique(b) # Handle ties
            if len(b) == 0:
                b = np.array([np.mean(D_obs[:, j])])
            bins.append(b)
            
            D_obs_disc[:, j] = np.digitize(D_obs[:, j], b)
            D_int_disc[:, j] = np.digitize(D_int[:, j], b)
            
        return D_obs_disc, D_int_disc

    def _bdeu_local_score(self, X_child, X_parents, K_child=5):
        """
        Dirichlet-Multinomial BDeu Marginal Log Likelihood (Eq 68 in BRCD Paper)
        """
        N = len(X_child)
        if X_parents.shape[1] == 0:
            # Single node
            q = 1
            alpha_ij = self.alpha_star / (q * K_child)
            alpha_i = self.alpha_star / q
            
            counts = np.bincount(X_child, minlength=K_child)
            score = gammaln(alpha_i) - gammaln(alpha_i + N) + np.sum(gammaln(alpha_ij + counts) - gammaln(alpha_ij))
            return float(score)
        else:
            # Group by parent configurations
            # Map parent state tuples to integer index
            par_tuples = [tuple(row) for row in X_parents]
            unique_pars, inverse_indices = np.unique(par_tuples, axis=0, return_inverse=True)
            q = len(unique_pars)
            alpha_ij = self.alpha_star / (q * K_child)
            alpha_i = self.alpha_star / q
            
            total_score = 0.0
            for u in range(q):
                mask = (inverse_indices == u)
                N_u = np.sum(mask)
                if N_u > 0:
                    counts = np.bincount(X_child[mask], minlength=K_child)
                    term = (gammaln(alpha_i) - gammaln(alpha_i + N_u) +
                            np.sum(gammaln(alpha_ij + counts) - gammaln(alpha_ij)))
                    total_score += term
                    
            return float(total_score)

    def fit_and_rank(self, D_obs, D_int):
        dag = self.system.dag
        D_obs_d, D_int_d = self._discretize(D_obs, D_int)
        
        # Candidate shift score: Difference in local BDeu score when F is added as parent to R
        scores = {}
        for r_idx in self.candidate_indices:
            parents = list(dag.predecessors(r_idx))
            X_par_int = D_int_d[:, parents]
            X_par_obs = D_obs_d[:, parents]
            
            # Score 1: R under invariant mechanism (combined)
            X_all = np.concatenate([D_obs_d[:, r_idx], D_int_d[:, r_idx]])
            if len(parents) == 0:
                X_par_all = np.empty((len(X_all), 0), dtype=int)
            else:
                X_par_all = np.vstack([X_par_obs, X_par_int])
                
            score_invariant = self._bdeu_local_score(X_all, X_par_all)
            
            # Score 2: R under separated interventional mechanism (F -> R)
            score_obs = self._bdeu_local_score(D_obs_d[:, r_idx], X_par_obs)
            score_int = self._bdeu_local_score(D_int_d[:, r_idx], X_par_int)
            score_shifted = score_obs + score_int
            
            # Log Bayes Factor for mechanism change at node R:
            # delta = Score(F -> R) - Score(F not -> R)
            delta_score = score_shifted - score_invariant
            scores[r_idx] = float(delta_score)

        ranking = [(self.node_names[r], scores[r]) for r in self.candidate_indices]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
