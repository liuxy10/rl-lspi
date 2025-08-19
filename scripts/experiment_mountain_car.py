import numpy as np

import lspi
import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym
from src.lspi.utils import vis_memory_visited_state, vis_best_action, vis_Q
import argparse

# compare to PPO baseline
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import numpy as np
from lspi.envs.cont_MountainCar import Continuous_MountainCarEnv





class CustomInitVelocityMountainCar(Continuous_MountainCarEnv):
    def __init__(self, init_position=None, init_velocity=0., ):

        self.init_position = init_position
        self.init_velocity = init_velocity
        super().__init__()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # Override position and velocity at reset
        if self.init_position is not None:
            pos =  self.init_position
        else:
            pos = obs[0]
        vel = self.init_velocity

        self.state = np.array([pos, vel], dtype=np.float32)
        return np.array(self.state, dtype=np.float32)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        dense_reward = -abs(self.goal_position - obs[0]) * 60
        reward += dense_reward
        # print("Reward dense = {}".format(dense_reward))
        
        return obs, reward, done, info



## sys args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='QuadraticAgent') #QuadraticAgent
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--memory_size', type=int, default=20000)
    parser.add_argument('--update_size', type=int, default=20000)
    return parser.parse_args()

args = parse_args()

# build the agent

if args.agent == 'PPO':
    # env = make_vec_env("MountainCarContinuous-v0", n_envs=1)
    def make_wrapper_env():
        env = CustomInitVelocityMountainCar(init_velocity=0.03)
        env.reset()
        return env
    
    

    env = DummyVecEnv([make_wrapper_env])
    model = PPO("MlpPolicy", env, verbose=0, n_steps= 1024, learning_rate=3e-4, batch_size=32, n_epochs=10)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Init Mean reward: {mean_reward} +/- {std_reward}")

    model.learn(total_timesteps=1e5)
    model.save("ppo_MountainCarContinuous")

    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_MountainCarContinuous")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"End Mean reward: {mean_reward} +/- {std_reward}")


elif args.agent == 'QuadraticAgent':
    def score(agent):
        episodes_reward, episode_lengths = lspi.utils.evaluate_policy(agent,
                                                        agent.env,
                                                        max_length=300,
                                                        n_eval_episodes=10)
        return np.mean(np.array(episodes_reward) / np.array(episode_lengths)),np.std(np.array(episodes_reward) / np.array(episode_lengths)), int(np.mean(episode_lengths))


    
    env = CustomInitVelocityMountainCar(init_velocity=0.03)

    agent = lspi.agents.QuadraticAgent(env)
    observation = env.reset() 

    # build the trainer
    gamma = 0.95
    memory_size = args.memory_size
    memory_type = 'sample'
    eval_type = 'iterative' ## TODO: try only using this type first
    baseline = lspi.baselines.LSPolicyIteration(env, agent, gamma, memory_size,
                                                memory_type, eval_type)

    # build the memory
    baseline.init_memory()
    print('memory size = {}'.format(len(baseline.memory)))
    vis_memory_visited_state(baseline)
    # vis_Q(baseline)

    # run the algorithm

    # run the algorithm for multiple times
    n_iter = 6
    n_reps = 3
    all_rewards = np.zeros((n_reps, n_iter + 1))

    for rep in range(n_reps):
        print(f"--- Repetition {rep + 1}/{n_reps} ---")
        rew, std, steps = score(agent)
        all_rewards[rep, 0] = rew
        print('iteration = {:02d} - average reward : {:.2f} +- {:.2f} '.
                format(0, rew, std))
        w = baseline.agent.weights.copy()
        for it in range(1, n_iter + 1):
            
            baseline.train_step()
            print("w diff = ", np.linalg.norm(baseline.agent.weights - w))
            w = baseline.agent.weights.copy()
            # vis_best_action(baseline)
            rew, std, steps = score(agent)
            all_rewards[rep, it] = rew
            # env.render() # Optional: rendering can slow down the process
            print('iteration = {:02d} - average reward : {:.2f} +- {:.2f} '.
                format(it, rew, std))


        # vis_best_action(baseline)
        baseline.init_memory(agent=agent, update_size=10)  # Re-initialize memory for the next repetition
        vis_memory_visited_state(baseline)
        vis_best_action(baseline)
        print("mean of memory:", np.mean([sample.r for sample in baseline.memory], axis=0))


    # Calculate mean and standard deviation across repetitions
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    iterations = np.arange(n_iter + 1)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_rewards, marker='o', linestyle='-', label='Mean Average Reward')
    plt.fill_between(iterations, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    alpha=0.2, label='Standard Deviation')
    plt.plot(all_rewards.T, color='gray', alpha=0.3, linewidth=0.5)  #  Individual runs
    plt.title('Average Reward vs. Iteration (3 Repetitions)')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.xticks(iterations)
    plt.grid(True)
    plt.legend()
    plt.show()

