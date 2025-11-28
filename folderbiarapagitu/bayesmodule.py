from probability import BayesNet, enumeration_ask, T, F

class GridBayesEngine:
    def __init__(self):
        self.bn = self._make_grid_bayes_net()
        
    def _make_grid_bayes_net(self):
        """
        Constructs the BayesNet using probability.py.
        Maps 3 Fuzzy States to 2 Boolean Variables.
        """
        return BayesNet([
            ('Risk_Elevated', '', 0.5), 
            ('Risk_Critical', '', 0.5), 
            
            # CPT: {(Elevated, Critical): Probability_of_Overload}
            ('Overload', 'Risk_Elevated Risk_Critical', {
                (T, T): 0.95,  # High Risk Context
                (T, F): 0.40,  # Medium Risk Context
                (F, F): 0.05,  # Low Risk Context
                (F, T): 0.95   # Invalid state, treat as High
            })
        ])

    def _map_fuzzy_to_evidence(self, score):
        """karena node di BN itu boolean, kita perlu mapping dari fuzzy score ke evidence boolean."""
        evidence = {}
        # Is it at least Medium?
        if score >= 40: evidence['Risk_Elevated'] = T
        else: evidence['Risk_Elevated'] = F
            
        # Is it High?
        if score >= 75: evidence['Risk_Critical'] = T
        else: evidence['Risk_Critical'] = F
            
        return evidence

    def get_failure_probability(self, fuzzy_score):
        """
        Main public method. Takes a score, returns probability (0.0 - 1.0).
        """
        # 1. Convert Score to Boolean Evidence
        evidence = self._map_fuzzy_to_evidence(fuzzy_score)
        
        # 2. Query the Network
        result_dist = enumeration_ask('Overload', evidence, self.bn)
        
        # 3. Return the probability of True
        return result_dist[T]