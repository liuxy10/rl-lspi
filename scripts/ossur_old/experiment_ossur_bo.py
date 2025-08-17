import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import lspi

class BayesianActionSelectionAgent:
    """
    An agent that uses Bayesian Optimization to select actions from a discrete set.
    It models the Q-function for the current state using a Gaussian Process
    and selects actions by maximizing the Upper Confidence Bound (UCB).
    """
    def __init__(self, env, gamma=0.95, kappa=2.576):
        self.env = env
        self.gamma = gamma
        self.kappa = kappa # Corresponds to 99% confidence interval
        self.history = [] # Stores (state, action, reward, next_state) tuples

        # Define the discrete 2D action space
        self.possible_actions = np.array([[i, j] for i in [-1, 0, 1] for j in [-1, 0, 1]])

        # A GP to model Q(s, a) for a fixed s. The action space is now 2D.
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True)

    def get_action(self, state):
        """
        Uses Bayesian Optimization to find the best discrete action for the given state.
        """
        # For the first few steps, explore randomly from the discrete action set
        if len(self.history) < len(self.possible_actions):
            action_idx = np.random.choice(len(self.possible_actions))
            return self.possible_actions[action_idx]

        # Prepare data for the GP: X is actions, y is target Q-values
        # Note: action is a 2D vector now
        X = np.array([exp[1] for exp in self.history])
        y = []
        for _, action, reward, next_state in self.history:
            # Estimate Q(s,a) using Bellman equation: r + gamma * max_a' Q(s', a')
            # To estimate max_a' Q(s', a'), we predict from our current GP over all possible next actions.
            q_next_state, _ = self.gp.predict(self.possible_actions, return_std=True)
            max_q_next = np.max(q_next_state)
            y.append(reward + self.gamma * max_q_next)

        self.gp.fit(X, np.array(y))

        # Calculate UCB for all possible actions to find the best one
        mean, std = self.gp.predict(self.possible_actions, return_std=True)
        ucb_scores = mean + self.kappa * std

        # Select the action with the highest UCB score
        best_action_index = np.argmax(ucb_scores)
        action = self.possible_actions[best_action_index]

        return action

    def update_history(self, state, action, reward, next_state):
        """Adds the latest experience to the history."""
        self.history.append((state, action, reward, next_state))


# --- Main Execution ---
if __name__ == '__main__':
    # Build the environment with a 2D action space
    nA = 2
    env = lspi.envs.SimulatorEnv(np.eye(nA))

    # Instantiate the agent
    agent = BayesianActionSelectionAgent(env, gamma=0.95, kappa=2.5)

    # Training and evaluation loop
    n_episodes = 20
    max_steps_per_episode = 200
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.update_history(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        print(f"Episode {episode + 1}/{n_episodes} - Length: {step + 1}, Reward: {total_reward:.2f}")

    print("\n--- Evaluation Finished ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
