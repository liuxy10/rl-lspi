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
   
    plt.figure(figsize=(8, 8))
    # if "Puddle" in baseline.env.spec.id:  # draw square (0,1) for puddle world
    #     plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='blue', alpha=0.5)
    for sample in baseline.memory:
        visited_states.append(sample.s)
        rewards.append(sample.r)
        dones.append(sample.done)
    plt.scatter(*zip(*visited_states), s=1, c=rewards)
    plt.colorbar()
    # mark end pos
    done_states = np.array(visited_states)[np.array(dones)]
    done_idx = np.where(np.array(dones))[0]
    start_idx = done_idx[:-1] + 1
    start_idx = [0] + start_idx.tolist()
    start_states = np.array(visited_states)[start_idx]
    # print(done_states, start_states)
    # assert len(done_states) == len(start_states), "Done states and start states should have the same length"
    plt.scatter(*zip(*done_states), s=10, c='red', marker="x", label='End States')
    plt.scatter(*zip(*start_states), s=10, c='green',marker="o",  label='Start States')
    plt.plot(*zip(*visited_states), alpha = 0.1)
    
    plt.title("Visited States")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    # if "Puddle" in baseline.env.spec.id:
    #     plt.axis("equal")
    plt.legend()
    plt.show()


def vis_Q(bl, n_grid = 30):
    num_actions = len(bl.env.actions)
    fig, axes = plt.subplots(1, num_actions, figsize=(5 * num_actions, 5), sharey=True)
    if num_actions == 1:
        axes = [axes]

    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    
    q_values = np.zeros((n_grid, n_grid, num_actions))

    for i in range(n_grid):
        for j in range(n_grid):
            state = np.array([grid_x[i, j], grid_y[i, j]])
            q_values[i, j, :] = bl.agent.get_q_values(state)
               

    vmin = q_values.min()
    vmax = q_values.max()

    for i, ax in enumerate(axes):
        im = ax.imshow(q_values[:, :, i].T, extent=(0, 1, 0, 1), origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Q(s,{bl.env.actions[i]})")
        ax.set_xlabel("X")
        if i == 0:
            ax.set_ylabel("Y")
        if "Puddle" in bl.env.spec.id:
            ax.set_aspect("equal")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    plt.suptitle("Q-Value Heatmaps for Each Action")
    plt.show()





def vis_best_action(bl, n_grid=30):
    num_actions = len(bl.env.actions)
    if num_actions == 0:
        print("No actions to visualize.")
        return

    # --- Grid and Q-value calculation ---
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    q_values = np.zeros((n_grid, n_grid, num_actions))
    pixels = np.zeros((n_grid, n_grid, 3))

    for i in range(n_grid):
        for j in range(n_grid):
            state = np.array([grid_x[i, j], grid_y[i, j]])
            q_values[i, j, :] = bl.agent.get_q_values(state)
            # For reward overlay
            pixels[j, i, :] = bl.env.get_reward(state)

    # --- Create Figure and Axes using GridSpec ---
    fig = plt.figure(figsize=(2 * num_actions, 8))
    gs = plt.GridSpec(2, num_actions, height_ratios=[1, 3])

    # --- Row 1: Q-Value Heatmaps ---
    q_axes = [fig.add_subplot(gs[0, i]) for i in range(num_actions)]
    vmin = q_values.min()
    vmax = q_values.max()

    for i, ax in enumerate(q_axes):
        im = ax.imshow(q_values[:, :, i].T, extent=(0, 1, 0, 1), origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Q(s, {bl.env.actions[i]})")
        ax.set_aspect("equal")
        if i > 0:
            ax.set_yticklabels([]) # Hide y-axis labels for subplots other than the first


    # fig.colorbar(im, ax=q_axes, orientation='vertical', fraction=0.1, pad=0.5)

    # --- Row 2: Best Action Visualization ---
    ax_best_action = fig.add_subplot(gs[1, :])
    best_action_indices = np.argmax(q_values, axis=2)
    actions_array = np.array(bl.env.actions)
    best_actions = actions_array[best_action_indices]

    ax_best_action.quiver(grid_x, grid_y, best_actions[..., 0] / n_grid * 5, best_actions[..., 1] / n_grid * 5,
                          angles='xy', scale_units='xy', scale=0.3, color='blue', alpha=0.5)

    # Normalize and overlay stage reward
    if pixels.max() > pixels.min():
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())
    
    ax_best_action.imshow(pixels, extent=(0, 1, 0, 1), origin='lower', alpha=0.5)
    ax_best_action.set_title("Best Action for Each Grid Point")
    ax_best_action.set_xlabel("X")
    ax_best_action.set_ylabel("Y")
    ax_best_action.set_aspect("equal")

    plt.suptitle("Policy and Q-Value Visualization", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()




