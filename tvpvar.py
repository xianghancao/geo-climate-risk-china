import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import invwishart, invgamma

# 1. Create a new TVPVAR class as a subclass of sm.tsa.statespace.MLEModel
class TVPVAR(sm.tsa.statespace.MLEModel):
    # Steps 2-3 are best done in the class "constructor", i.e. the __init__ method
    def __init__(self, y):
        # Create a matrix with [y_t' : y_{t-1}'] for t = 2, ..., T
        augmented = sm.tsa.lagmat(y, 1, trim='both', original='in', use_pandas=True)
        # Separate into y_t and z_t = [1 : y_{t-1}']
        p = y.shape[1]
        y_t = augmented.iloc[:, :p]
        z_t = sm.add_constant(augmented.iloc[:, p:])

        # Recall that the length of the state vector is p * (p + 1)
        k_states = p * (p + 1)
        super().__init__(y_t, exog=z_t, k_states=k_states)

        # Note that the state space system matrices default to contain zeros,
        # so we don't need to explicitly set c_t = d_t = 0.

        # Construct the design matrix Z_t
        # Notes:
        # -> self.k_endog = p is the dimension of the observed vector
        # -> self.k_states = p * (p + 1) is the dimension of the observed vector
        # -> self.nobs = T is the number of observations in y_t
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog + 1)
            end = start + self.k_endog + 1
            self['design', i, start:end, :] = z_t.T

        # Construct the transition matrix T = I
        self['transition'] = np.eye(k_states)

        # Construct the selection matrix R = I
        self['selection'] = np.eye(k_states)

        # Step 3: Initialize the state vector as alpha_1 ~ N(0, 5I)
        self.ssm.initialize('known', stationary_cov=5 * np.eye(self.k_states))

    # Step 4. Create a method that we can call to update H and Q
    def update_variances(self, obs_cov, state_cov_diag):
        self['obs_cov'] = obs_cov
        self['state_cov'] = np.diag(state_cov_diag)

    # Finally, it can be convenient to define human-readable names for
    # each element of the state vector. These will be available in output
    @property
    def state_names(self):
        state_names = np.empty((self.k_endog, self.k_endog + 1), dtype=object)
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names[i] = (
                ['intercept.%s' % endog_name] +
                ['L1.%s->%s' % (other_name, endog_name) for other_name in self.endog_names])
        return state_names.ravel().tolist()
