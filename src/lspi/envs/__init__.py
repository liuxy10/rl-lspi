from lspi.envs.chain_walk import ChainWalkEnv
from lspi.envs.inv_pendulum import InvertedPendulumEnv
from lspi.envs.bound_linear import SimulatorEnv
from lspi.envs.puddle_env import PuddleEnv

from gym.envs.registration import register

register(
    id='ChainWalk-v0',
    entry_point='lspi.envs:ChainWalkEnv',
    max_episode_steps=1000
)

register(
    id='LSPI-InvertedPendulum-v0',
    entry_point='lspi.envs:InvertedPendulumEnv',
    max_episode_steps=1000
)

register(
    id='BoundLinear-v0',
    entry_point='lspi.envs:SimulatorEnv',
    max_episode_steps=1000
)

register(
    id='LSPI-Puddle-v0',
    entry_point='lspi.envs:PuddleEnv',
    max_episode_steps=500
)
