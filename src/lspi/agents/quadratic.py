
from qpsolvers import solve_qp
import numpy as np
from lspi.agents.agent import Agent
import os
from scipy import sparse
class QuadraticAgent(Agent):
    """
    Quadratic Agent: 
    features: upper triangular matrix of the outer product of state and action
    action: gym.box
    """
    def __init__(self, env, w=None, preprocess_obs=lambda x: x):
        obs_len = len(env.observation_space.sample())
        act_len = len(env.action_space.sample()) if type(env.action_space.sample()) is np.ndarray else 1
        self.n = obs_len + act_len
        self.features_size = self.n * (self.n + 1) // 2 
        
        super(QuadraticAgent, self).__init__(env, preprocess_obs)
        if w is not None:
            assert w.shape[0] == self.features_size, \
                "w should have shape ({},), got {}".format(self.features_size, w.shape)
            self.set_weights(w)
        else:
            self.init_weights()
        
    def init_weights(self, scale=1.):
        S = np.diag(np.random.random( size=self.n))
        self.weights = - self.convertS2W(S)
        assert len(self.weights) == self.features_size, "weights should have shape ({},), got {}".format(self.features_size, self.weights.shape)

    def get_features_size(self):
        obs = self.env.observation_space.sample()
        act = self.env.action_space.sample()
        features = self.get_features(obs, act)
        return len(features)


    def _get_features(self, obs, action):
        sa = np.concatenate((obs, action))
        phi = np.outer(sa, sa)[np.triu_indices(self.n)]
        assert len(phi) == self.features_size
        return phi

    def get_q_gradient(self, obs, action):
        obs = self.preprocess_obs(obs)
        return self._get_q_gradient(obs, action)

    def _get_q_gradient(self, obs, action): 
        """ q func gradient w.r.t each action analytical soln """
        out = np.zeros(self.n)
        ## TODO
        pass

    def predict(self, obs, eps= 0.1):
        H = -self.convertW2S(self.weights)
        R = sparse.csc_matrix(H[len(obs):, len(obs):])
        Hax = H[:len(obs), len(obs):]
        obs = self.preprocess_obs(obs)
        ub = self.env.action_space.high
        
        # efficiently solve the Quadratic program argmin_a 0 + a^T R a + 2 * a^T Hax x using
        action = solve_qp(R, Hax.T @ obs,
                 None, None, # no inequality constraints
                 None, None, # no equality constraints
                lb = -ub, ub = ub,
                solver='osqp')
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        # print(f"Predicted action: {action}")

        if np.random.rand() < eps:
            action = self.env.action_space.sample()
        return action

    def convertW2S(self, w):
        """Convert weight vector to a symmetric matrix."""
        Phat = np.zeros((self.n, self.n))
        Phat[np.triu_indices(self.n)] = w
        S = (Phat + Phat.T)
        S[np.diag_indices(self.n)] /= 2 # ensure diagonal is not added twice
        return S

    def convertS2W(self, S):
        """Convert symmetric matrix to weight vector."""
        assert S.shape == (self.n, self.n)
        w = S[np.triu_indices(self.n)]
        return w
    
    def save(self, folder_path, name = "weights_online" ):
         # save the weights
        os.makedirs(folder_path, exist_ok=True)
        if ".npz" not in name:
            name += ".npz"

        print(f"Saving weights to {os.path.join(folder_path, name)}")
        np.savez(os.path.join(folder_path, name ), weights=self.weights)

    

