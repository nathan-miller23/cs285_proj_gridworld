from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class MyEnv(MyMiniGridEnv):

    def __init__(self, size=9, max_steps=100, start_pos=(1, 1), good_goal_pos=(7, 8), bad_goal_pos=(8, 7), reward='sparse', good_goal_reward=10, bad_goal_reward=-10, gamma=1.0):
        self.start_pos = start_pos
        self.good_goal_pos = good_goal_pos
        self.bad_goal_pos = bad_goal_pos
        self.reward = reward
        self.good_goal_reward = good_goal_reward
        self.bad_goal_reward = bad_goal_reward
        self.gamma = gamma
        super(MyEnv, self).__init__(grid_size=size, max_steps=max_steps)


    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(GoodGoal(), *self.good_goal_pos)
        self.put_obj(BadGoal(), *self.bad_goal_pos)

        self.agent_pos = self.start_pos
        self.agent_dir = 0

        self.mission = "Be the best agent I can be"

    def _reward(self):
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell.goal_type == 'good':
            return (self.gamma**self.step_count) * self.good_goal_reward
        elif curr_cell.goal_type == 'bad':
            return (self.gamma**self.step_count) * self.bad_goal_reward
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

    def get_accessible_states(self):
        empty_states = self.get_empty_states()
        empty_states.append(self.good_goal_pos)
        empty_states.append(self.bad_goal_pos)
        return empty_states


    

