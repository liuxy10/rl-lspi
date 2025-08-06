import numpy as np

import lspi

def score(agent):
    _, episode_lengths = lspi.utils.evaluate_policy(agent,
                                                    agent.env,
                                                    max_length=3000,
                                                    n_eval_episodes=10,
                                                    vis=False)
    return int(np.mean(episode_lengths))

# build the environment
nA = 3
env = lspi.envs.InvertedPendulumEnv(nA)

# build the agent
grids = [[-np.pi / 4., 0., np.pi / 4], [-1., 0., 1.]]
centers = lspi.agents.RadialAgent.get_centers_from_grids(grids)
sigma = 1.
agent = lspi.agents.RadialAgent(env, centers, sigma)

# build the trainer
gamma = 0.95
memory_size = 1000
memory_type = 'episode'
eval_type = 'batch'
baseline = lspi.baselines.LSPolicyIteration(env, agent, gamma, memory_size,
                                            memory_type, eval_type)

# build the memory
baseline.init_memory()
print('memory size = {}'.format(len(baseline.memory)))

# run the algorithm
n_iter = 10
steps = score(agent)
print('iteration = {:02d} - average number of balancing steps : {:04d}'.format(
    0, steps))
for it in range(1, n_iter + 1):
    baseline.train_step()
    steps = score(agent)
    print('iteration = {:02d} - average number of balancing steps : {:04d}'.
          format(it, steps))