"""
    2D ergodic control formulated as Heat Equation Driven Area Coverage (HEDAC) objective,
	with a spatial distribution described as a mixture of Gaussians.
 
    Copyright (c) 2023 Idiap Research Institute, https://www.idiap.ch
    Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

    This file is part of RCFS <https://rcfs.ch>
    License: GPL-3.0-only
"""

import numpy as np
import matplotlib.pyplot as plt

# Helper class
# ===============================
class SecondOrderAgent:
    """
    A point mass agent with second order dynamics.
    """
    def __init__(
        self,
        x,
        nbDataPoints,
        max_dx=1,
        max_ddx=0.2,
    ):
        self.x = np.array(x)  # position
        # determine which dimesnion we are in from given position
        self.nbVarX = len(x)
        self.dx = np.zeros(self.nbVarX)  # velocity

        self.t = 0  # time
        self.dt = 1  # time step
        self.nbDatapoints = nbDataPoints

        self.max_dx = max_dx
        self.max_ddx = max_ddx

        # we will store the actual and desired position
        # of the agent over the timesteps
        self.x_arr = np.zeros((self.nbDatapoints, self.nbVarX))
        self.des_x_arr = np.zeros((self.nbDatapoints, self.nbVarX))

    def update(self, gradient):
        """
        set the acceleration of the agent to clamped gradient
        compute the position at t+1 based on clamped acceleration
        and velocity
        """
        ddx = gradient # we use gradient of the potential field as acceleration
        # clamp acceleration if needed
        if np.linalg.norm(ddx) > self.max_ddx:
            ddx = self.max_ddx * ddx / np.linalg.norm(ddx)

        self.x = self.x + self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        self.x_arr[self.t] = np.copy(self.x)
        self.t += 1

        self.dx += self.dt * ddx  # compute the velocity
        # clamp velocity if needed
        if np.linalg.norm(self.dx) > self.max_dx:
            self.dx = self.max_dx * self.dx / np.linalg.norm(self.dx)

# Helper functions for HEDAC
# ===============================
def rbf(mean, x, eps):
    """
    Radial basis function w/ Gaussian Kernel
    """
    d = x - mean  # radial distance
    l2_norm_squared = np.dot(d, d)
    # eps is the shape parameter that can be interpreted as the inverse of the radius
    return np.exp(-eps * l2_norm_squared)


def normalize_mat(mat):
    return mat / (np.sum(mat) + 1e-10)


def calculate_gradient(agent, gradient_x, gradient_y):
    """
    Calculate movement direction of the agent by considering the gradient
    of the temperature field near the agent
    """
    # find agent pos on the grid as integer indices
    adjusted_position = agent.x / param.dx
    # note x axis corresponds to col and y axis corresponds to row
    col, row = adjusted_position.astype(int)


    gradient = np.zeros(2)
    # if agent is inside the grid, interpolate the gradient for agent position
    if row > 0 and row < param.height - 1 and col > 0 and col < param.width - 1:
        gradient[0] = bilinear_interpolation(gradient_x, adjusted_position)
        gradient[1] = bilinear_interpolation(gradient_y, adjusted_position)

    # if kernel around the agent is outside the grid,
    # use the gradient to direct the agent inside the grid
    boundary_gradient = 2  # 0.1
    pad = param.kernel_size - 1
    if row <= pad:
        gradient[1] = boundary_gradient
    elif row >= param.height - 1 - pad:
        gradient[1] = -boundary_gradient

    if col <= pad:
        gradient[0] = boundary_gradient
    elif col >= param.width - pad:
        gradient[0] = -boundary_gradient

    return gradient


def clamp_kernel_1d(x, low_lim, high_lim, kernel_size):
    """
    A function to calculate the start and end indices
    of the kernel around the agent that is inside the grid
    i.e. clamp the kernel by the grid boundaries
    """
    start_kernel = low_lim
    start_grid = x - (kernel_size // 2)
    num_kernel = kernel_size
    # bound the agent to be inside the grid
    if x <= -(kernel_size // 2):
        x = -(kernel_size // 2) + 1
    elif x >= high_lim + (kernel_size // 2):
        x = high_lim + (kernel_size // 2) - 1

    # if agent kernel around the agent is outside the grid,
    # clamp the kernel by the grid boundaries
    if start_grid < low_lim:
        start_kernel = kernel_size // 2 - x - 1
        num_kernel = kernel_size - start_kernel - 1
        start_grid = low_lim
    elif start_grid + kernel_size >= high_lim:
        num_kernel -= x - (high_lim - num_kernel // 2 - 1)
    if num_kernel > low_lim:
        grid_indices = slice(start_grid, start_grid + num_kernel)

    return grid_indices, start_kernel, num_kernel


def agent_block(min_val, agent_radius):
    """
    A matrix representing the shape of an agent (e.g, RBF with Gaussian kernel). 
    min_val is the upper bound on the minimum value of the agent block.
    """
    nbVarX = 2  # number of dimensions of space

    eps = 1.0 / agent_radius  # shape parameter of the RBF
    l2_sqrd = (
        -np.log(min_val) / eps
    )  # squared maximum distance from the center of the agent block
    l2_sqrd_single = (
        l2_sqrd / nbVarX
    )  # maximum squared distance on a single axis since sum of all axes equal to l2_sqrd
    l2_single = np.sqrt(l2_sqrd_single)  # maximum distance on a single axis
    # round to the nearest larger integer
    if l2_single.is_integer(): 
        l2_upper = int(l2_single)
    else:
        l2_upper = int(l2_single) + 1
    # agent block is symmetric about the center
    num_rows = l2_upper * 2 + 1
    num_cols = num_rows
    block = np.zeros((num_rows, num_cols))
    center = np.array([num_rows // 2, num_cols // 2])
    for i in range(num_rows):
        for j in range(num_cols):
            block[i, j] = rbf(np.array([j, i]), center, eps)
    # we hope this value is close to zero 
    print(f"Minimum element of the block: {np.min(block)}" +
          " values smaller than this assumed as zero")
    return block


def offset(mat, i, j):
    """
    offset a 2D matrix by i, j
    """
    rows, cols = mat.shape
    rows = rows - 2
    cols = cols - 2
    return mat[1 + i : 1 + i + rows, 1 + j : 1 + j + cols]


def border_interpolate(x, length, border_type):
    """
    Helper function to interpolate border values based on the border type
    (gives the functionality of cv2.borderInterpolate function)
    """
    if border_type == "reflect101":
        if x < 0:
            return -x
        elif x >= length:
            return 2 * length - x - 2
    return x


def bilinear_interpolation(grid, pos):
    """
    Linear interpolating function on a 2-D grid
    """
    x, y = pos.astype(int)
    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1], "reflect101")
    x1 = border_interpolate(x + 1, grid.shape[1], "reflect101")
    y0 = border_interpolate(y, grid.shape[0], "reflect101")
    y1 = border_interpolate(y + 1, grid.shape[0], "reflect101")
    # Distance from lower integers
    xd = pos[0] - x0
    yd = pos[1] - y0
    # Interpolate on x-axis
    c01 = grid[y0, x0] * (1 - xd) + grid[y0, x1] * xd
    c11 = grid[y1, x0] * (1 - xd) + grid[y1, x1] * xd
    # Interpolate on y-axis
    c = c01 * (1 - yd) + c11 * yd
    return c


# Helper functions borrowed from SMC example given in 
# demo_ergodicControl_2D_01.py for using the same 
# target distribution and comparing the results
# of SMC and HEDAC
# ===============================
def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
    # Base case: A Hadamard matrix of size 1 is just [[1]].
    if n == 1:
        return np.array([[1]])

    # Recursively construct a Hadamard matrix of size n/2.
    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    # Combine the four sub-matrices to form a Hadamard matrix of size n.
    h = np.empty((n, n), dtype=int)
    h[:half_size,:half_size] = h_half
    h[half_size:,:half_size] = h_half
    h[:half_size:,half_size:] = h_half
    h[half_size:,half_size:] = -h_half

    return h


def get_GMM(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Gaussian centers
    Mu1 = [0.6, 0.7]
    Mu2 = [0.6, 0.3]
    # Gaussian covariances
    # direction vectors for constructing the covariance matrix using
    # outer product of a vector with itself then the principal direction
    # of covariance matrix becomes the given vector and its orthogonal
    # complement
    Sigma1_v = [0.3, 0.1]
    Sigma2_v = [0.1, 0.2]
    # scale
    Sigma1_scale = 5e-1
    Sigma2_scale = 3e-1
    # regularization
    Sigma1_regularization = np.eye(param.nbVarX) * 5e-3
    Sigma2_regularization = np.eye(param.nbVarX) * 1e-2
    # GMM Gaussian Mixture Model

    # Gaussian centers
    Mu = np.zeros((param.nbVarX, param.nbGaussian))
    Mu[:, 0] = np.array(Mu1)
    Mu[:, 1] = np.array(Mu2)
    # covariance matrices
    Sigma = np.zeros((param.nbVarX, param.nbVarX, param.nbGaussian))
    # construct the covariance matrix using the outer product
    Sigma[:, :, 0] = (
        np.vstack(Sigma1_v) @ np.vstack(Sigma1_v).T * Sigma1_scale
        + Sigma1_regularization
    )
    Sigma[:, :, 1] = (
        np.vstack(Sigma2_v) @ np.vstack(Sigma2_v).T * Sigma2_scale
        + Sigma2_regularization
    )
    # mixing. coefficients Priors (summing to one)
    Alpha = (
        np.ones(param.nbGaussian) / param.nbGaussian
    )
    return Mu, Sigma, Alpha


def discrete_gmm(param):
    """
    Same GMM as in ergodic_control_SMC.py
    """
    # Discretize given GMM using Fourier basis functions
    rg = np.arange(0, param.nbFct, dtype=float)
    KX = np.zeros((param.nbVarX, param.nbFct, param.nbFct))
    KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
    # Mind the flatten() !!!

    # Explicit description of w_hat by exploiting the Fourier transform
    # properties of Gaussians (optimized version by exploiting symmetries)
    op = hadamard_matrix(2 ** (param.nbVarX - 1))
    op = np.array(op)
    # check the reshaping dimension !!!
    kk = KX.reshape(param.nbVarX, param.nbFct**2) * param.omega

    # Compute fourier basis function weights w_hat for the target distribution given by GMM
    w_hat = np.zeros(param.nbFct**param.nbVarX)
    for j in range(param.nbGaussian):
        for n in range(op.shape[1]):
            MuTmp = np.diag(op[:, n]) @ param.Mu[:, j]
            SigmaTmp = np.diag(op[:, n]) @ param.Sigma[:, :, j] @ np.diag(op[:, n]).T
            cos_term = np.cos(kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
            # Eq.(22) where D=1
            w_hat = w_hat + param.Alpha[j] * cos_term * exp_term
    w_hat = w_hat / (param.L**param.nbVarX) / (op.shape[1])

    # Fourier basis functions (for a discretized map)
    xm1d = np.linspace(param.xlim[0], param.xlim[1], param.nbRes)  # Spatial range
    xm = np.zeros((param.nbGaussian, param.nbRes, param.nbRes))
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
    # Mind the flatten() !!!
    ang1 = (
        KX[0, :, :].flatten().T[:, np.newaxis]
        @ xm[0, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    ang2 = (
        KX[1, :, :].flatten().T[:, np.newaxis]
        @ xm[1, :, :].flatten()[:, np.newaxis].T
        * param.omega
    )
    phim = np.cos(ang1) * np.cos(ang2) * 2 ** (param.nbVarX)
    # Some weird +1, -1 due to 0 index !!!
    xx, yy = np.meshgrid(np.arange(1, param.nbFct + 1), np.arange(1, param.nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(param.nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
    phim = phim * np.tile(HK, (param.nbRes**param.nbVarX, 1)).T

    # Desired spatial distribution
    g = w_hat.T @ phim
    return g


# Parameters
# ===============================
param = lambda: None # Lazy way to define an empty class in python
param.nbDataPoints = 400
param.min_kernel_val = 1e-8  # upper bound on the minimum value of the kernel
param.diffusion = 3  # increases global behavior
param.source_strength = 1  # increases local behavior
param.obstacle_strength = 0  # increases local behavior
param.agent_radius =15  # changes the effect of the agent on the coverage
param.max_dx = 1 # maximum velocity of the agent
param.max_ddx = 0.1 # maximum acceleration of the agent
param.cooling_radius = (
    1  # changes the effect of the agent on local cooling (collision avoidance)
)
param.nbAgents = 1 # number of agents
param.local_cooling = 0  # for multi agent collision avoidance
param.dx = 1

param.nbVarX = 2  # dimension of the space
param.nbResX = 100 # number of grid cells in x direction
param.nbResY = 100 # number of grid cells in y direction

param.nbGaussian = 2#2

param.nbFct = 10 # Number of basis functions along x and y
# Domain limit for each dimension (considered to be 1
# for each dimension in this implementation)
param.xlim = [0, 1]
param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
param.omega = 2 * np.pi / param.L

param.nbRes = param.nbResX  # resolution of discretization

param.alpha = np.array([1, 1]) * param.diffusion

# G = np.zeros((param.nbResX, param.nbResY))

# # Note this part is needed to have exact same target distribution as in ergodic_control_SMC.py
# # param.Mu, param.Sigma, param.Alpha = get_fixed_GMM(param)
# param.Mu, param.Sigma, param.Alpha = get_GMM(param)


# g = discrete_gmm(param)
# G = np.reshape(g, [param.nbResX, param.nbResY])
# G = np.abs(G)  # there is no negative heat

import gym
from gym.envs.registration import register
register(
    id='LSPI-Puddle-v0',
    entry_point='lspi.envs:PuddleEnv',
    max_episode_steps=400,
)
env = gym.make('LSPI-Puddle-v0')


G = np.zeros((param.nbResX, param.nbResY))

for i in range(param.nbResX):
    for j in range(param.nbResY):
        x = float(i)/param.nbResX
        y = float(j)/param.nbResY
        G[j,i] = env.get_reward(np.array([x,y]))

# Initialize agents
# ===============================
agents = []
for i in range(param.nbAgents):
    # initial position of the agent
    # x0 = np.random.uniform(0, param.nbResX, 2)
    x0 = env.reset()*100 #np.array([10,30]) # if single agent same ic as SMC example
    agent = SecondOrderAgent(x=x0, nbDataPoints=param.nbDataPoints,max_dx=param.max_dx,max_ddx=param.max_ddx)
    # agent = FirstOrderAgent(x=x, dim_t=cfg.timesteps)
    rgb = np.random.uniform(0, 1, 3)
    agent.color = np.concatenate((rgb, [1.0]))  # append alpha value
    agents.append(agent)

# Initialize heat equation related fields
# ===============================
# precompute everything we can before entering the loop
coverage_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))
heat_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))
local_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))
goal_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))

param.height, param.width = G.shape

param.area = param.dx * param.width * param.dx * param.height

goal_density = normalize_mat(G)

coverage_density = np.zeros((param.height, param.width))
heat = np.array(goal_density)

max_diffusion = np.max(param.alpha)
param.dt = min(
    1.0, (param.dx * param.dx) / (4.0 * max_diffusion)
)  # for the stability of implicit integration of Heat Equation
coverage_block = agent_block(param.min_kernel_val, param.agent_radius)
cooling_block = agent_block(param.min_kernel_val, param.cooling_radius)
param.kernel_size = coverage_block.shape[0]


# HEDAC Loop
# ===============================
# do absolute minimum inside the loop for speed
for t in range(param.nbDataPoints):
    # cooling of all the agents for a single timestep
    # this is used for collision avoidance bw/ agents
    local_cooling = np.zeros((param.height, param.width))
    for agent in agents:
        # find agent pos on the grid as integer indices
        p = agent.x
        adjusted_position = p / param.dx
        col, row = adjusted_position.astype(int)

        # each agent has a kernel around it,
        # clamp the kernel by the grid boundaries
        row_indices, row_start_kernel, num_kernel_rows = clamp_kernel_1d(
            row, 0, param.height, param.kernel_size
        )
        col_indices, col_start_kernel, num_kernel_cols = clamp_kernel_1d(
            col, 0, param.width, param.kernel_size
        )

        # add the kernel to the coverage density
        # effect of the agent on the coverage density
        coverage_density[row_indices, col_indices] += coverage_block[
            row_start_kernel : row_start_kernel + num_kernel_rows,
            col_start_kernel : col_start_kernel + num_kernel_cols,
        ]

        # local cooling is used for collision avoidance between the agents
        # so it can be disabled for speed if not required
        # if param.local_cooling != 0:
        #     local_cooling[row_indices, col_indices] += cooling_block[
        #         row_start_kernel : row_start_kernel + num_kernel_rows,
        #         col_start_kernel : col_start_kernel + num_kernel_cols,
        #     ]
        # local_cooling = normalize_mat(local_cooling)

    coverage = normalize_mat(coverage_density)

    # this is the part we introduce exploration problem to the Heat Equation
    diff = goal_density - coverage
    sign = np.sign(diff)
    source = np.maximum(diff, 0) ** 2
    source = normalize_mat(source) * param.area

    current_heat = np.zeros((param.height, param.width))

    # 2-D heat equation (Partial Differential Equation)
    # In 2-D we perform this second-order central for x and y.
    # Note that, delta_x = delta_y = h since we have a uniform grid.
    # Accordingly we have -4.0 of the center element.

    # At boundary we have Neumann boundary conditions which assumes
    # that the derivative is zero at the boundary. This is equivalent
    # to having a zero flux boundary condition or perfect insulation.
    current_heat[1:-1, 1:-1] = param.dt * (
        (
            +param.alpha[0] * offset(heat, 1, 0)
            + param.alpha[0] * offset(heat, -1, 0)
            + param.alpha[1] * offset(heat, 0, 1)
            + param.alpha[1] * offset(heat, 0, -1)
            - 4.0 * offset(heat, 0, 0)
        )
        / (param.dx * param.dx)
        + param.source_strength * offset(source, 0, 0)
        - param.local_cooling * offset(local_cooling, 0, 0)
    ) + offset(heat, 0, 0)

    heat = current_heat.astype(np.float32)

    # Calculate the first derivatives mind the order x and y
    gradient_y, gradient_x = np.gradient(heat, 1, 1)

    for agent in agents:
        grad = calculate_gradient(
            agent,
            gradient_x,
            gradient_y,
        )
        local_heat = bilinear_interpolation(current_heat, agent.x)
        agent.update(grad)

    coverage_arr[..., t] = coverage
    heat_arr[..., t] = heat

# Plot
# ===============================
fig, ax = plt.subplots(1, 3, figsize=(16, 8))

ax[0].set_title("Agent trajectory and desired GMM")
# Required for plotting discretized GMM
xlim_min = 0
xlim_max = param.nbResX
xm1d = np.linspace(xlim_min, xlim_max, param.nbResX)  # Spatial range
xm = np.zeros((param.nbGaussian, param.nbResX, param.nbResY))
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)
X = np.squeeze(xm[0, :, :])
Y = np.squeeze(xm[1, :, :])

ax[0].contourf(X, Y, G, cmap="gray_r") # plot discrete GMM
# Plot agent trajectories
for agent in agents:
    ax[0].plot(
        agent.x_arr[0, 0], agent.x_arr[0, 1], marker=".", color="black", markersize=10
    )
    ax[0].plot(
        agent.x_arr[:, 0],
        agent.x_arr[:, 1],
        color="black",
        linewidth=1,
    )
ax[0].set_aspect("equal", "box")

ax[1].set_title("Exploration goal (heat source), explored regions at time t")
arr = goal_density - coverage_arr[..., -1]
arr_pos = np.where(arr > 0, arr, 0) 
arr_neg = np.where(arr < 0, -arr, 0)
ax[1].contourf(X, Y, arr_pos,cmap='gray_r')
# Plot agent trajectories
for agent in agents:
    ax[1].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linewidth=10, color="blue",label="agent footprint") # sensor footprint
    ax[1].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linestyle="--", color="black",label='agent path') # trajectory line
ax[1].legend(loc="upper left")
ax[1].set_aspect("equal", "box")

ax[2].set_title("Gradient of the potential field")
gradient_y, gradient_x = np.gradient(heat_arr[..., -1])
ax[2].quiver(X, Y, gradient_x, gradient_y,scale= 20,units='xy') # Scales the length of the arrow inversely
# ax[2].quiver(X, Y, gradient_x, gradient_y)

# Plot agent trajectories
for agent in agents:
    ax[2].plot(agent.x_arr[:, 0], agent.x_arr[:, 1], linestyle="--", color="black") # trajectory line
    ax[2].plot(
        agent.x_arr[0, 0], agent.x_arr[0, 1], marker=".", color="black", markersize=10
    )
ax[2].set_aspect("equal", "box")

plt.show()
