import numpy as np


class Agent:
    def __init__(self, env, preprocess_obs=None):
        if preprocess_obs is None:
            preprocess_obs = lambda x: x

        self.env = env
        self.action_size = self.env.action_space.shape[0] if self.env.action_space.__class__.__name__ == "Box" else self.env.action_space.n
        self.preprocess_obs = preprocess_obs
        self.features_size = self.get_features_size()
        self.init_weights()

    def init_weights(self, scale=1.): # for discrete action only
        size = self.features_size * self.action_size 
        self.weights = np.random.normal(size=size, scale=scale)

    def set_weights(self, weights):
        self.weights = weights

    def get_features_size(self):
        obs = self.env.observation_space.sample()
        features = self.get_features(obs)
        return len(features)

    def get_features(self, obs, action = None):
        obs = self.preprocess_obs(obs)
        return self._get_features(obs, action) if self.__class__.__name__ == "QuadraticAgent" else self._get_features(obs)

    def _get_features(self, obs, action = None):
        pass


    def predict(self, obs, eps = 0.0):
        values = np.dot(
            self.weights.reshape(self.action_size, self.features_size),
            self.get_features(obs))
        action = np.argmax(values)
        if np.random.rand() < eps:
            action = np.random.randint(self.action_size)
        return action

