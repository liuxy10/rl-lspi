import itertools

import numpy as np
from lspi.agents.agent import Agent


class RadialAgent(Agent):
    def __init__(self, env, centers, sigma=1., remap = True, preprocess_obs=None):
        self.centers = centers
        self.sigma2 = sigma**2
        self.remap = remap
        if remap: # remap from 0-1 to low to high of obs space
            self.centers = self.centers * (env.observation_space.high - env.observation_space.low) + env.observation_space.low
            self.sigma2 = self.sigma2 * ((env.observation_space.high - env.observation_space.low)**2)
        super(RadialAgent, self).__init__(env, preprocess_obs)

    def _get_features(self, obs):
        dists = np.power(self.centers - obs, 2)
        if self.remap:
            rbfs = np.exp(-np.sum(dists / (2 * self.sigma2), axis=1))
        else:
            rbfs = np.exp(-dists.sum(1) / (2 * self.sigma2)) 
        return np.append(rbfs, [1.])
    
    def _get_feature_gradient(self,obs):
        dists = np.power(self.centers - obs, 2)
        if self.remap:
            rbfs = np.exp(-np.sum(dists / (2 * self.sigma2), axis=1))
        else:
            rbfs = np.exp(-dists.sum(1) / (2 * self.sigma2))
        grad = -(rbfs / (2 * self.sigma2)[:, np.newaxis]).T * (self.centers - obs)
        return np.vstack((grad, [0., 0.]))

    def get_q_values(self, obs): # discrete only 
        return np.dot(
            self.weights.reshape(self.action_size, self.features_size),
            self.get_features(obs)).copy()
    
    @staticmethod
    def get_centers_from_grids(grids):
        return np.array(list(itertools.product(*grids)))
    

    def get_q_gradient(self, obs):
        obs = self.preprocess_obs(obs)
        return self._get_q_gradient(obs)
    
    def _get_q_gradient(self, obs):
        # finite difference in obs space
        
        grad = np.zeros([len(obs), self.action_size])
        for i in range(len(obs)):
            for delta in [-1, 1]:
                obs_delta = obs.copy()
                obs_delta[i] += delta * 1e-5
                grad[i] += (self.get_q_values(obs_delta) 
                            - self.get_q_values(obs))/2e-5 * delta
        return grad.copy()

    def _get_q_gradient_analytical(self, obs):
        # analytical # TODO fix this
        grad = np.zeros([len(obs), self.action_size])
        for a in range(self.action_size):
            grad[:, a] =  (self.weights[a * self.features_size:(a + 1) * self.features_size][np.newaxis,:] @ self._get_feature_gradient(obs)).ravel()
        assert np.allclose(grad, self._get_q_gradient_fd(obs), atol=1e-2), f"gradients not match with finite difference result, {grad} != {self._get_q_gradient_fd(obs)} (FD)"
        return grad

    ## TODO Unproved: Predict action based on gradient, comment it if not wanna use
    # def predict(self, obs, eps = 0.1):
    #     obs = self.preprocess_obs(obs)
    #     grad = self.get_q_gradient(obs).mean(axis = 1) # mean across actions
        
    #     grad_dir = grad / np.linalg.norm(grad, ord=1) if np.linalg.norm(grad, ord=1) > 0 else grad
    #     acts = np.diag(np.sign(grad_dir))
    #     actions = [np.zeros(2) for i in range(5)] # HACK: follow action definition in puddle world
    #     for i in range(4):
    #         actions[i][i//2] = 1 * (i%2 * 2 - 1)
    #     if np.random.uniform(0, 1) < eps: return np.random.randint(0, 5)
    #     p = np.random.uniform(0, 1)
    #     if p < abs(grad_dir[0]):
    #         return np.where([np.all(action == acts[0]) for action in actions])[0][0]
    #     else:
    #         return np.where([np.all(action == acts[1]) for action in actions])[0][0]
        
        