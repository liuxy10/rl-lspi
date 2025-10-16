import math
import random

# import gym
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding



class SimulatorEnv(gym.Env):
    """    A simple simulator environment for testing LSPI agents with a linear model."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, A, e_out=0.0, e_in=0.0, 
                 input_scale=1., input_offset=0, 
                 output_scale=1., output_offset=0,
                 Q=None, R=None, bound=None, verbose=False):
        super(SimulatorEnv, self).__init__()

        self.A = A
        self.input_scale = np.array(input_scale * np.ones(A.shape[1]), dtype=np.float64)
        self.output_scale = np.array(output_scale * np.ones(A.shape[0]), dtype=np.float64)
        self.input_offset = np.array(input_offset * np.ones(A.shape[1]), dtype=np.float64)
        self.output_offset = np.array(output_offset * np.ones(A.shape[0]), dtype=np.float64)
        self.e_out = e_out
        self.e_in = e_in
        self.n_action = self.A.shape[1]
        self.n_state = self.A.shape[0]
        

        if bound is not None:
            self.param_bound = bound
        else:
            lim = 3 * self.input_scale
            low = self.input_offset - lim
            high = self.input_offset + lim
            self.param_bound = np.stack([low, high], axis=-1)

        self.Q = Q if Q is not None else np.diag(np.ones(self.n_state)/self.n_state)
        self.R = R if R is not None else np.zeros((self.n_action, self.n_action))

        
        # define action space
        self.actions = np.array([0.5 * np.linspace(-1., 1., 3) for _ in range(self.n_action)]).T
        self.action_space = spaces.MultiDiscrete([3] * self.n_action) if self.n_action > 1 else spaces.Discrete(3)  # 3 actions for each parameter
        # observation_space: parameters/outputs
        obs_low = np.full((self.n_state,), -3)
        obs_high = np.full((self.n_state,), 3)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.params = np.zeros(self.n_action)
        self.verbose = verbose
        self.dt = 0.1

        self.state = None  # internal current "observation" (output of predict)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def step(self, action):
        # First order integrator: param_{t+1} = param_t + delta_action
        
        u = np.array([self.actions[a, i] for i, a in enumerate(action)]) if isinstance(action, (list, np.ndarray)) else self.actions[action]
        # u += 0.1 * np.random.uniform(-1, 1)
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        delta = u / self.input_scale
        next_params = self.params + delta * self.dt  # 0.1 is the time step, adjust as needed TODO

        # Bound check in normalized (internal) space
        bound = (self.param_bound - self.input_offset[..., None]) / self.input_scale[..., None]
        next_params = np.clip(next_params, bound[:, 0], bound[:, 1])

        self.params = next_params

        obs = self.predict()
        print("Action:", action, "Params:", self.params, "Obs:", obs) if self.verbose else None
        self.state = obs
        # You must define your "goal state" externally and pass target as argument if needed
        reward = -self._calc_stage_cost(obs, np.zeros_like(obs), action)  # negative cost as reward

        done = np.any(np.abs(self.params - self.param_bound[..., 0]) < 1e-3) | np.any(np.abs(self.params - self.param_bound[..., 1]) < 1e-3) # fail if the parameter is at edge of the bound
        truncated = False # truncated if you want to limit the episode length
        info = {}

        return obs.copy(), reward, done, truncated, info
    
    def reset(self, *args, seed=None, options=None, **kwargs):
        self.params = np.zeros(self.n_action) + np.random.randn(self.n_action) * 1
        self.state = self.predict()
        return self.state.copy()

    def predict(self):
        # Predict next state based on current params
        params_with_noise = self.params + self.e_in * np.random.randn(self.n_action)
        pred = self.A @ (params_with_noise - self.input_offset) / self.input_scale
        return self.e_out * np.random.randn(self.n_state) + pred * self.output_scale + self.output_offset

    def _calc_stage_cost(self, y, target, action=None):
        Q = self.Q
        R = self.R
        y = np.asarray(y)
        target = np.asarray(target)
        diff = y - target
        if diff.ndim == 1:
            diff = diff.reshape(1, -1)
        elif diff.shape[-1] == self.n_state:
            diff = diff.reshape(-1, self.n_state)
        elif diff.shape[0] == self.n_state:
            diff = diff.T
        state_cost = 0.5 * np.einsum('bi,ij,bj->b', diff, Q, diff)
        if action is not None:
            action = np.asarray(action)
            if action.ndim == 1:
                action = action.reshape(1, -1)
            elif action.shape[-1] == self.n_action:
                action = action.reshape(-1, self.n_action)
            elif action.shape[0] == self.n_action:
                action = action.T
            action_cost = 0.5 * np.einsum('bi,ij,bj->b', action, R, action)
        else:
            action_cost = 0.0
        total_cost = state_cost + action_cost
        return total_cost[0] if total_cost.shape[0] == 1 else total_cost

    def render(self, mode="human"):
        print("Params:", self.get_params())
        print("State:", self.state)

    def close(self):
        pass

    def get_params(self):
        return self.params.copy() * self.input_scale + self.input_offset
    




