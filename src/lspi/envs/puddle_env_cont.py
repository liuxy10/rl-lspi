import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
from gym.envs.classic_control import rendering
from gym_puddle.shapes.image import Image
import pyglet
import gym
from gym.envs.classic_control import rendering

class PuddleContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, start=[0.7, 0.4],
                  goal=[0.8, 0.8], goal_threshold=0.1,
            noise=0.02, thrust=0.05, puddle_center=[
                # [.3, .6], 
                # [.4, .5], 
                # [.8, .9]
                ],
            puddle_width=[
                # [.1, .05], 
                # [.04, .1], 
                # [.08, .12]
                ], max_episode_steps=500, punish_bound = True):
        self.start = np.array(start) if start is not None else None
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.noise = noise
        self.thrust = thrust
        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_width = [np.array(width) for width in puddle_width]

        self.action_space = spaces.Box(low=np.array([-1, -1]) * thrust, high=np.array([1, 1]) * thrust, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.4,0.0]), high=np.array([0.75, 1.0]), dtype=np.float32)

        # self.actions = [np.zeros(2) for i in range(5)]
        # for i in range(4):
        #     self.actions[i][i//2] = thrust * (i%2 * 2 - 1)

        self.seed()
        self.counter = 0
        self.viewer = None
        self.his_obs = []
        self.max_episode_steps = max_episode_steps
        self.punish_bound = punish_bound 
        if punish_bound:
            low = self.observation_space.low
            high = self.observation_space.high
            n_bound_pts = 8
            # bottom edge
            bound = np.array([np.linspace(low[0], high[0], n_bound_pts), np.ones(n_bound_pts) * low[1]]).T
            self.puddle_center += bound.tolist()
            # top edge
            bound = np.array([np.linspace(low[0], high[0], n_bound_pts), np.ones(n_bound_pts) * high[1]]).T
            self.puddle_center += bound.tolist()
            # left edge
            bound = np.array([np.ones(n_bound_pts) * low[0], np.linspace(low[1], high[1], n_bound_pts)]).T
            self.puddle_center += bound.tolist()
            # right edge
            bound = np.array([np.ones(n_bound_pts) * high[0], np.linspace(low[1], high[1], n_bound_pts)]).T
            self.puddle_center += bound.tolist()

            self.puddle_width += [[0.08 * (high[0] - low[0]), 0.08 * (high[1] - low[1])] for _ in range(n_bound_pts*4)]
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.pos += action + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, self.observation_space.low, self.observation_space.high)
        
        reward = self.get_reward(self.pos)
        self.counter += 1
 
        done = self.counter >= self.max_episode_steps # np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold or 
        # update history
        self.his_obs.append(copy.copy(self.pos))
        self.his_obs = self.his_obs[-self.max_episode_steps//5:]
        return self.pos, reward, done, {}

    def get_reward(self, pos):
        reward = -1.
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            reward -= 5. * self._gaussian1d(pos[0], cen[0], wid[0]) * \
                self._gaussian1d(pos[1], cen[1], wid[1])

        # reward -= 30. * np.linalg.norm(pos - self.goal, ord=1)  # distance to goal
        reward += 40. * self._gaussian1d(pos[0], self.goal[0], self.goal_threshold * 3) * \
                  self._gaussian1d(pos[1], self.goal[1], self.goal_threshold*3)
        if np.linalg.norm(pos - self.goal, ord=1) < self.goal_threshold:
            reward += 50.  # bonus for reaching the goal

        return reward

    def _gaussian1d(self, p, mu, sig):
        return np.exp(-((p - mu)**2)/(2.*sig**2)) / (sig*np.sqrt(2.*np.pi))

    def reset(self):
        self.counter = 0
        if self.start is None:
            self.pos = self.observation_space.sample()
        else:
            self.pos = copy.copy(self.start)
        return self.pos

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400

        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)

            
            img_width = 100
            img_height = 100
            fformat = 'RGB'
            pixels = np.zeros((img_width, img_height, len(fformat)))

            for i in range(img_width):
                for j in range(img_height):
                    x = float(i)/img_width
                    y = float(j)/img_height
                    pixels[j,i,:] = self.get_reward(np.array([x,y]))
            

            pixels -= pixels.min()
            pixels *= 255./pixels.max()
            pixels = np.floor(pixels)

            img = pyglet.image.create(img_width, img_height)
            img.format = fformat
            data=[chr(int(pixel)) for pixel in pixels.flatten()]

            img.set_data(fformat, img_width * len(fformat), ''.join(data))
            bg_image = Image(img, screen_width, screen_height)
            bg_image.set_color(1.0,1.0,1.0)

            self.viewer.add_geom(bg_image)

            thickness = 5
            agent_polygon = rendering.FilledPolygon([(-thickness,-thickness),
             (-thickness,thickness), (thickness,thickness), (thickness,-thickness)])
            agent_polygon.set_color(0.0,1.0,0.0)
            self.agenttrans = rendering.Transform()
            agent_polygon.add_attr(self.agenttrans)
            self.viewer.add_geom(agent_polygon)

             # Visualize start point
            if self.start is not None:
                print("Start point:", self.start)
                start_geom = rendering.make_circle(5)
                start_geom.set_color(0,0,1) # Blue
                start_trans = rendering.Transform(translation=(self.start[0]*screen_width, self.start[1]*screen_height))
                start_geom.add_attr(start_trans)
                self.viewer.add_geom(start_geom)

            # Visualize goal point
            if self.goal is not None:
                print("Goal point:", self.goal)
                goal_geom = rendering.make_circle(5)
                goal_geom.set_color(1,0,0) # Red
                goal_trans = rendering.Transform(translation=(self.goal[0]*screen_width, self.goal[1]*screen_height))
                goal_geom.add_attr(goal_trans)
                self.viewer.add_geom(goal_geom)

        self.agenttrans.set_translation(self.pos[0]*screen_width, self.pos[1]*screen_height)


        counter_text = f"Counter: {self.counter}"
        label = pyglet.text.Label(counter_text,
                              font_size=12,
                              x=10, y=screen_height - 20,
                              anchor_x='left', anchor_y='top',
                              color=(255, 255, 255, 255))
        self.viewer.render(return_rgb_array = mode=='rgb_array')
        label.draw()          



class Simulate(PuddleContinuousEnv):
    # what is different is that we define the map from our data, not from gaussian1ds. 
    def __init__(self, start=None, # TODO: can only support 2d param space now
                  goal=None, 
                  goal_threshold=0.1,
                  noise_a=0.02,
                  noise_f=0.0,
                  thrust=0.08,
                  feature_map=None,
                  reward_map=None, max_episode_steps = 100):
        super().__init__(start=start, goal=goal, goal_threshold=goal_threshold,
                         noise=noise_a, thrust=thrust, max_episode_steps=max_episode_steps)
        self._feature_map = feature_map
        self._reward_map = reward_map
        self.noise_f = noise_f
        self.f = np.zeros(self._feature_map.n_features)  # feature vector
        assert self._feature_map.n_features == len(self.goal), "Feature map and goal must have the same number of features"

    def step(self, action):
        # Apply the feature map to the action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.pos += action + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)
        reward = None
        # map the pos to feature space
        if self._feature_map is not None:
            self.f = self._feature_map(self.pos) + self.np_random.uniform(low=-self.noise_f, high=self.noise_f, size=(self._feature_map.n_features,))
        if self._reward_map is not None:
            reward = self.get_reward(self.pos, action)
        self.counter += 1
        done = self.counter >= self.max_episode_steps
        # update history
        self.his_obs.append(copy.copy(self.pos))
        self.his_obs = self.his_obs[-self.max_episode_steps//5:] # just for vis
        return self.pos, reward, done, {'feature': self.f, 'history': self.his_obs}

    def get_reward(self, state, action = None, goal = None):
        reward = -1.
        for cen, wid in zip(self.puddle_center, self.puddle_width):
            reward -= 2. * self._gaussian1d(state[0], cen[0], wid[0]) * \
                self._gaussian1d(state[1], cen[1], wid[1])
        f = self._feature_map(state)
        goal = self.goal if goal is None else goal
        return self._reward_map(f, action, goal) if self._reward_map is not None else None
