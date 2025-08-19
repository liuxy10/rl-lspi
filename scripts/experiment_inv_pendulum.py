import numpy as np

import lspi
import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lspi.utils import vis_memory_visited_state, vis_best_action, vis_Q
import argparse
## sys args
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='RadialAgent')
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--update_size', type=int, default=100)
    return parser.parse_args()

def score(agent):
    episodes_reward, episode_lengths = lspi.utils.evaluate_policy(agent,
                                                    agent.env,
                                                    max_length=300,
                                                    n_eval_episodes=10)
    return np.mean(np.array(episodes_reward) / np.array(episode_lengths)),np.std(np.array(episodes_reward) / np.array(episode_lengths)), int(np.mean(episode_lengths))


# parse the args
args = parse_args()
# build the environment
nA = 3
env = lspi.envs.InvertedPendulumEnv(nA)

# build the agent
if args.agent == 'RadialAgent':
    grids = [[-np.pi / 4., 0., np.pi / 4], [-1., 0., 1.]]
    centers = lspi.agents.RadialAgent.get_centers_from_grids(grids)
    sigma = 1.
    agent = lspi.agents.RadialAgent(env, centers, sigma)
else: 
    raise ValueError("Unknown/ unimplemented agent type: {}".format(args.agent))

# build the trainer
gamma = 0.95
memory_size = args.memory_size
memory_type = 'episode'
eval_type = 'batch'
baseline = lspi.baselines.LSPolicyIteration(env, agent, gamma, memory_size,
                                            memory_type, eval_type)

# build the memory
baseline.init_memory()
print('memory size = {}'.format(len(baseline.memory)))
vis_memory_visited_state(baseline)
vis_Q(baseline)

# run the algorithm
n_iter = args.n_iter
steps = score(agent)
print('iteration = {:02d} - average number of balancing steps : {:04d}'.format(
    0, steps))
w = baseline.agent.weights.copy()
for it in range(1, n_iter + 1):
    baseline.train_step()
    print("w diff = ", np.linalg.norm(baseline.agent.weights - w))
    w = baseline.agent.weights.copy()
    steps = score(agent)
    
    print('iteration = {:02d} - average number of balancing steps : {:04d}'.
          format(it, steps))

# baseline.init_memory(baseline.agent, update_size=10)
# vis_memory_visited_state(baseline)
vis_Q(baseline)