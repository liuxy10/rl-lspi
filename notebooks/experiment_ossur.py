import numpy as np

import lspi

def score(agent):
    episode_reswards, episode_lengths = lspi.utils.evaluate_policy(agent,
                                                    agent.env,
                                                    max_length=3000,
                                                    n_eval_episodes=10, 
                                                    vis=True)

    return np.mean(np.array(episode_reswards)/np.array(episode_lengths)),int(np.mean(episode_lengths))

# build the environment
nA = 1
env = lspi.envs.SimulatorEnv(np.eye(nA))

# build the agent
grids = [np.linspace(env.observation_space.low[i], env.observation_space.high[i], 3) for i in range(nA)]
centers = lspi.agents.RadialAgent.get_centers_from_grids(grids)
sigma = .5
agent = lspi.agents.RadialAgent(env, centers, sigma)


# build the trainer
gamma = 0.95
memory_size = 3000
memory_type = 'episode'
eval_type = 'batch'
baseline = lspi.baselines.LSPolicyIteration(env, agent, gamma, memory_size,
                                            memory_type, eval_type)

# build the memory
baseline.init_memory()
print('memory size = {}'.format(len(baseline.memory)))

# run the algorithm
n_iter = 10
rews, steps = score(agent)
print('iteration = {:02d} - average number of balancing steps : {:04d}'.format(
    0, steps))
for it in range(1, n_iter + 1):
    baseline.train_step()
    rews, steps = score(agent)
    print('iteration = {:02d} - average reward : {:.2f} - average number of balancing steps : {:04d}'.
          format(it, rews, steps))

