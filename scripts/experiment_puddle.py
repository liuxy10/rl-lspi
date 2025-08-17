import numpy as np
import gym
import gym, gym_puddle # Don't forget this extra line!
import lspi

import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lspi.utils import vis_memory_visited_state, vis_best_action

from gym.envs.registration import register
def score(agent):
    episodes_reward, episode_lengths = lspi.utils.evaluate_policy(agent,
                                                    agent.env,
                                                    max_length=500,
                                                    n_eval_episodes=10,
                                                    vis = False)
    return np.mean(np.array(episodes_reward) / np.array(episode_lengths)),np.std(np.array(episodes_reward) / np.array(episode_lengths)), int(np.mean(episode_lengths))


register(
    id='LSPI-Puddle-v0',
    entry_point='lspi.envs:PuddleEnv',
    max_episode_steps=500
)
env = gym.make('LSPI-Puddle-v0')
env.start = None

# build the agent
grids = [[0.,0.25, 0.5,0.25, 1.], [0.,0.25, 0.5,0.25, 1.]]
centers = lspi.agents.RadialAgent.get_centers_from_grids(grids)
sigma = .2
agent = lspi.agents.RadialAgent(env, centers, sigma)

# build the trainer
gamma = 0.95
memory_size = 10
memory_type = 'episode'
eval_type = 'sherman_morrison' # 'iterative' # 'batch'


# run the algorithm for multiple times
n_iter = 6
n_reps = 3
all_rewards = np.zeros((n_reps, n_iter + 1))

for rep in range(n_reps):
    print(f"--- Repetition {rep + 1}/{n_reps} ---")
    # Re-initialize agent and baseline for a fresh start
    agent = lspi.agents.RadialAgent(env, centers, sigma)
    baseline = lspi.baselines.LSPolicyIteration(env, agent, gamma, memory_size,
                                                memory_type, eval_type)
    baseline.init_memory()
    print("mean of memory:", np.mean([sample.r for sample in baseline.memory], axis=0))

    # Visualize the visited states in memory
    vis_memory_visited_state(baseline)
    vis_best_action(baseline)

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


    vis_best_action(baseline)
    baseline.init_memory(agent=agent, update_size=10)  # Re-initialize memory for the next repetition
    vis_memory_visited_state(baseline)
    print("mean of memory:", np.mean([sample.r for sample in baseline.memory], axis=0))
    # delete cache and variables in this run
    del baseline
    del agent
    # del env

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

