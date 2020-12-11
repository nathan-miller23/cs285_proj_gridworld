import random

from gym_minigrid.minigrid import *


class RandomPoints(MyMiniGridEnv):

    def __init__(self, size=9, max_steps=100, start_pos=(1, 1), good_goal_pos=(7, 7), reward='sparse',
                 good_goal_reward=10, bad_goal_reward=-10, num_bad_goals=5, num_points=5, gamma=1.0):
        self.start_pos = start_pos
        self.good_goal_pos = good_goal_pos
        self.reward = reward
        self.good_goal_reward = good_goal_reward
        self.bad_goal_reward = bad_goal_reward
        self.num_bad_goals = num_bad_goals
        self.num_points = num_points
        self.gamma = gamma
        super(RandomPoints, self).__init__(grid_size=size, max_steps=max_steps)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        points = [(1, 1), self.good_goal_pos]
        self.put_obj(GoodGoal(), *self.good_goal_pos)
        for _ in range(self.num_points):
            prop_loc = (random.randint(1, width - 2), random.randint(1, height - 2))
            while prop_loc in points:
                prop_loc = (random.randint(1, width - 2), random.randint(1, height - 2))
            self.grid.wall_rect(prop_loc[0], prop_loc[1], 1, 1)
            points.append(prop_loc)

        for _ in range(self.num_bad_goals):
            prop_loc = (random.randint(1, width - 2), random.randint(1, height - 2))
            while prop_loc in points:
                prop_loc = (random.randint(1, width - 2), random.randint(1, height - 2))
            self.put_obj(BadGoal(), prop_loc[0], prop_loc[1])
            points.append(prop_loc)

        self.agent_pos = self.start_pos
        self.agent_dir = 0

        self.mission = "Be the best agent I can be"

    def _reward(self):
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell.goal_type == 'good':
            return (self.gamma ** self.step_count) * self.good_goal_reward
        elif curr_cell.goal_type == 'bad':
            return (self.gamma ** self.step_count) * self.bad_goal_reward
        else:
            raise ValueError("Called `self._reward()` at incorrect time!")

    def dist_to_goal(self, pos):
        x, y = pos
        goal_x, goal_y = self.good_goal_pos
        return abs(goal_x - x) + abs(goal_y - y)

    def _dense_reward(self, s, s_prime):
        if self.reward == 'sparse':
            return 0
        return self.dist_to_goal(s) - self.dist_to_goal(s_prime)
