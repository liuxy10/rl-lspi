import numpy as np
import gymnasium as gym

import lspi
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Register the custom environment
gym.register(
    id='MySimulatorEnv-v0',
    entry_point='lspi.envs:SimulatorEnv',
    kwargs={'A': np.eye(1)} # Assuming a 1-dimensional state/action space for simplicity
)


# build the environment
# The lspi.envs.SimulatorEnv needs to be compatible with the gymnasium.Env interface
# for this to work. It must have a Box observation space and a Discrete or Box action space for PPO.
# We assume lspi.envs.SimulatorEnv is compatible.
env = gym.make('MySimulatorEnv-v0')

# build the agent using SB3 PPO
# Note: PPO requires the environment to be registered if you pass its ID as a string.
# The action space for the default SimulatorEnv might need adjustment depending on its implementation.
# PPO supports Box, Discrete, MultiDiscrete, and MultiBinary action spaces.
train = False
if train:
    model = PPO("MlpPolicy", env, gamma=0.9, verbose=1)

    # train the agent
    model.learn(total_timesteps=100_000, log_interval=4)

    # save the agent
    model.save("ppo_SimEnv")

    del model # remove to demonstrate saving and loading

model = PPO.load("ppo_SimEnv")

# evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# You can also watch the agent play if the environment supports rendering
obs, info = env.reset()
print(f"Initial observation: {obs}")
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render() # This will fail if the env does not have a render method
    print(f"obs: {obs}, action: {action}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")
    if terminated or truncated:
        obs, info = env.reset()

