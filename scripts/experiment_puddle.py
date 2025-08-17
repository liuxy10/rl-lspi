import numpy as np
import gym
import gym, gym_puddle # Don't forget this extra line!
import lspi

import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lspi.utils import vis_memory_visited_state


def score(agent):
    episodes_reward, episode_lengths = lspi.utils.evaluate_policy(agent,
                                                    agent.env,
                                                    max_length=500,
                                                    n_eval_episodes=10,
                                                    vis = False)
    return np.mean(episodes_reward), int(np.mean(episode_lengths))

env = gym.make('PuddleWorld-v0')


def vis_best_action(bl, n_grid = 40):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    # for each grid point, visualize the best action
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, n_grid+1), np.linspace(0, 1, n_grid+1))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    actions = np.array([bl.env.actions[bl.agent.predict(pt)] for pt in grid_points])
    plt.quiver(grid_x, grid_y, actions[:, 0]/n_grid * 5, actions[:, 1]/n_grid * 5,
               angles='xy', scale_units='xy', scale=0.3, color='blue', alpha=0.5)
    pixels = np.zeros((n_grid, n_grid, 3))
    # overlay stage reward
    for i in range(n_grid):
        for j in range(n_grid):
            x = float(i)/n_grid
            y = float(j)/n_grid
            pixels[j,i,:] = bl.env.get_reward(np.array([x,y]))
    pixels -= pixels.min()
    pixels *=1./pixels.max()
    # pixels = np.floor(pixels)
    
    plt.imshow(pixels, extent=(0, 1, 0, 1), origin='lower', alpha=0.5)
    plt.title("Best Action for Each Grid Point")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()

# build the agent
grids = [[0.,0.25, 0.5,0.25, 1.], [0.,0.25, 0.5,0.25, 1.]]
centers = lspi.agents.RadialAgent.get_centers_from_grids(grids)
sigma = .2
agent = lspi.agents.RadialAgent(env, centers, sigma)

# build the trainer
gamma = 0.95
memory_size = 10
memory_type = 'episode'
eval_type = 'sherman_morrison' #'batch'


# run the algorithm for multiple times
n_iter = 5
n_reps = 5
all_rewards = np.zeros((n_reps, n_iter + 1))

for rep in range(n_reps):
    print(f"--- Repetition {rep + 1}/{n_reps} ---")
    # Re-initialize agent and baseline for a fresh start
    agent = lspi.agents.RadialAgent(env, centers, sigma)
    baseline = lspi.baselines.LSPolicyIteration(env, agent, gamma, memory_size,
                                                memory_type, eval_type)
    baseline.init_memory()
    # Visualize the visited states in memory
    # vis_memory_visited_state(baseline)

    rew, steps = score(agent)
    all_rewards[rep, 0] = rew
    print('iteration = {:02d} - average reward : {:.2f} - average number of steps : {:04d}'.
          format(0, rew, steps))

    for it in range(1, n_iter + 1):
        baseline.train_step()
        vis_best_action(baseline)
        rew, steps = score(agent)
        all_rewards[rep, it] = rew
        # env.render() # Optional: rendering can slow down the process
        print('iteration = {:02d} - average reward : {:.2f} - average number of steps : {:04d}'.
              format(it, rew, steps))
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

