import matplotlib.pyplot as plt
import numpy as np
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
    visited_states, rewards, dones = [], [], []
    # draw square (0,1)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='blue', alpha=0.5)
    for sample in baseline.memory:
        visited_states.append(sample.s)
        rewards.append(sample.r)
        dones.append(sample.done)
    plt.scatter(*zip(*visited_states), s=1, c=rewards)
    
    # mark end pos
    done_states = np.array(visited_states)[np.array(dones)]
    plt.scatter(*zip(*done_states), s=10, c='red', label='End States')
    plt.plot(*zip(*visited_states), alpha = 0.1)
    plt.colorbar()
    plt.title("Visited States")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.axis('equal')
    plt.show()


def vis_best_action(bl, n_grid = 30):
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




