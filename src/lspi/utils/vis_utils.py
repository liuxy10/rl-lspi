import matplotlib.pyplot as plt
def plot_state_trajectory(ep, state_traj, max_length = 50, savefigto=None):
    plt.figure()
    skip_rate = 1
    if state_traj.shape[0] > max_length:
        skip_rate = state_traj.shape[0] // max_length
        state_traj = state_traj[::skip_rate]
    if state_traj.ndim == 2 and state_traj.shape[1] >= 2:
        plt.plot(state_traj[:, 0], state_traj[:, 1])
        # Add arrows to show direction
        for i in range(len(state_traj) - 1):
            plt.arrow(state_traj[i, 0], state_traj[i, 1],
                      state_traj[i + 1, 0] - state_traj[i, 0],
                      state_traj[i + 1, 1] - state_traj[i, 1],
                      head_width=0.05, head_length=0.1, 
                      fc='blue', ec='blue')

        # mark end point with a different color
        plt.scatter(state_traj[-1, 0], state_traj[-1, 1], color='red', label='End Point')
        # mark desired end point (0,0) as a green dot
        plt.scatter(0, 0, color='green', label='Desired End Point')
        plt.xlabel('State dim 0')
        plt.ylabel('State dim 1')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.title(f'Episode {ep+1} State Trajectory, skip rate {skip_rate}')
        # equal axis scaling
        # plt.axis('equal')
        plt.legend()
    else:
        plt.plot(state_traj, marker='o')
        plt.xlabel('Step')
        plt.ylabel('State')
        plt.title(f'Episode {ep+1} State Trajectory')
    plt.show()

def vis_memory_visited_state(baseline):
    import matplotlib.pyplot as plt
    visited_states, rewards = [], []
    # draw square (0,1)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='blue', alpha=0.5)
    for sample in baseline.memory:
        visited_states.append(sample.s)
        rewards.append(sample.r)
    plt.scatter(*zip(*visited_states), s=1, c=rewards)
    plt.plot(*zip(*visited_states), alpha = 0.1)
    plt.colorbar()
    plt.title("Visited States")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.axis('equal')
    plt.show()



