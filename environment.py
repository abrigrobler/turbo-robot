#Preliminary: imports
import random
from utils import Utils

"""
gw.py is a modified version of gridworld.py by Anson Wong. The idea is to take his original 
environment and agent and modify the code in such a way that it suits my own experimentation.

Original header:

-------------------

 gridworld.py  (author: Anson Wong / git: ankonzoid)

 We use Q-learning to train an epsilon-greedy agent to find the shortest path 
 between position (0, 0) to opposing corner (Ny-1, Nx-1) of a 2D rectangular grid
 in the 2D GridWorld environment of size (Ny, Nx).

 Note: 
 The optimal policy exists but is a highly degenerate solution because
 of the multitude of ways one can traverse down the grid in the minimum
 number of steps. Therefore a greedy policy that always moves the agent closer 
 towards the goal can be considered an optimal policy (can get to the goal 
 in `Ny + Nx - 2` actions). In our example, this corresponds to actions 
 of moving right or down to the bottom-right corner.

 Example optimal policy:
 
  [[1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 2]
  [1 1 1 1 1 0]]

  action['up'] = 0
  action['right'] = 1
  action['down'] = 2
  action['left'] = 3 

"""
import random
import numpy as np


class Environment:

    def __init__(self, goal, invalid_states=[], Ny=8, Nx=8, r_goal=100, r_nongoal=-1):
        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        self.state = (0, 0)
        self.invalid_states = invalid_states
        self.goal_state = (goal[0], goal[1])
        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        # Define rewards table
        self.r_goal = r_goal
        self.r_nongoal = r_nongoal
        self.R = self._build_rewards()  # R(s,a) agent rewards
        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def print_layout(self):
        for y in range(self.Ny):
            for x in range(self.Nx):
                if (y, x) in self.invalid_states:
                    print('[█]', end='')
                elif (y, x) == self.goal_state:
                    print('[G]', end='')
                else:
                    print('[ ]', end='')
            print("")

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (0, 0)
        return self.state

    def generate_start(self):

        self.state = (random.randrange(self.Ny), random.randrange(self.Nx))

        while self.state == self.goal_state or self.state in self.invalid_states:
            self.state = (random.randrange(self.Ny), random.randrange(self.Nx))

        return self.state

    def step(self, action):
        # Evolve agent state
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        # Collect reward
        reward = self.R[self.state + (action,)]
        # Terminate if we reach bottom-right grid corner
        done = self.goal_state == self.state
        # Update state
        self.state = state_next
        return state_next, reward, done

    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]

        if (y > 0) and not (y - 1, x) in self.invalid_states:  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])

        if (y < self.Ny - 1) and not (y + 1, x) in self.invalid_states:  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])

        if (x > 0) and not (y, x - 1) in self.invalid_states:  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])

        if (x < self.Nx - 1) and not (y, x + 1) in self.invalid_states:  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])

        actions_allowed = np.array(actions_allowed, dtype=int)
        #print('State: {}, allowed actions: {}'.format(self.state, actions_allowed))
        return actions_allowed

    def get_allowed_actions_by_state(self, state):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = state[0], state[1]

        if (y > 0) and not (y - 1, x) in self.invalid_states:  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])

        if (y < self.Ny - 1) and not (y + 1, x) in self.invalid_states:  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])

        if (x > 0) and not (y, x - 1) in self.invalid_states:  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])

        if (x < self.Nx - 1) and not (y, x + 1) in self.invalid_states:  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])

        actions_allowed = np.array(actions_allowed, dtype=int)
        #print('State: {}, allowed actions: {}'.format(self.state, actions_allowed))
        return actions_allowed

    def _build_rewards(self):
        R = self.r_nongoal * \
            np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]

        if self.goal_state[0] - 1 > 0:
            R[self.goal_state[0] - 1, self.goal_state[1],
                self.action_dict["down"]] = self.r_goal  # arrive from above
        if self.goal_state[0] + 1 < self.Ny:
            R[self.goal_state[0] + 1, self.goal_state[1],
                self.action_dict["up"]] = self.r_goal  # arrive from below
        if self.goal_state[1] + 1 < self.Nx:
            R[self.goal_state[0], self.goal_state[1] + 1,
                self.action_dict["left"]] = self.r_goal  # arrive from right
        if self.goal_state[1] - 1 > 0:
            R[self.goal_state[0], self.goal_state[1] - 1,
                self.action_dict["right"]] = self.r_goal  # arrive from left

        return R


class FullEnvironment(Environment):
    def __init__(self, goal, invalid_states=[], Ny=8, Nx=8, r_goal=100, r_nongoal=-0.1):

        # Define state space
        self.Ny = Ny  # y grid size
        self.Nx = Nx  # x grid size
        self.state_dim = (Ny, Nx)
        self.state = (0, 0)
        self.invalid_states = invalid_states
        self.goal_state = (goal[0], goal[1])

        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations

        # Define rewards table
        self.r_goal = r_goal
        self.r_nongoal = r_nongoal
        self.R = self._build_rewards()  # R(s,a) agent rewards

        # Define distributed rewards table
        self.Z = np.zeros((Ny, Nx), dtype=int)
        self._populate_z()

        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

        # Create an empty list that will be filled with sub environments
        self.sub_envs = []

    def _get_distance_from_goal(self, s):
        dist_y = np.abs(self.goal_state[0] - s[0])
        dist_x = np.abs(self.goal_state[1] - s[1])
        return (dist_y**2 + dist_x**2)**0.5

    def _populate_z(self):
        for y in range(self.Ny):
            for x in range(self.Nx):
                if (y, x) == self.goal_state:
                    self.Z[y, x] = self.r_goal
                else:
                    self.Z[y, x] = self.r_goal / \
                        (self._get_distance_from_goal((y, x)) + 1)

    def update_z(self, q_list):
        Q = Utils.build_Q(np.array(q_list))
        for y in range(self.Ny):
            for x in range(self.Nx):
                self.Z[y, x] = (np.max(Q[y, x, :]) + self.Z[y, x])/2.0
        for r in self.sub_envs:
            r.set_z(self.Z)

    def set_sub_envs(self, ranges):
        for r in ranges:
            self.sub_envs.append(Region(self.goal_state, self.invalid_states,
                                        self.Ny, self.Nx, r[0], r[1], self.r_goal, self.r_nongoal, self.Z))


class Region(Environment):
    def __init__(self, goal, invalid_states, Ny, Nx, Ry, Rx, r_goal, r_nongoal, Z):
        # Ry, Rx are ranges with lower bound included, upper bound included
        self.goal_state = goal
        self.invalid_states = invalid_states
        self.state_dim = (Ny, Nx)
        self.r_goal = r_goal
        self.r_nongoal = r_nongoal
        self.Ry = (Ry[0], Ry[1] + 1)
        self.Rx = (Rx[0], Rx[1] + 1)
        self.Z = Z

        # Also save the size
        self.Ny = Ny
        self.Nx = Nx

        self.on_top_edge = self.Ry[0] == 0
        self.on_bottom_edge = self.Ry[1] == Ny

        self.on_left_edge = self.Rx[0] == 0
        self.on_right_edge = self.Rx[1] == Nx

        # Define action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations

        self.R = self._build_rewards()  # R(s,a) agent rewards

        # Check action space consistency
        if len(self.action_dict.keys()) != len(self.action_coords):
            exit("err: inconsistent actions given")

    def reset(self):
        # Reset agent state to top-left grid corner
        self.state = (self.Ry[0], self.Rx[0])
        return self.state

    def set_z(self, Z):
        self.Z = Z

    def generate_start(self):

        self.state = (random.randrange(
            self.Ry[0], self.Ry[1]), random.randrange(self.Rx[0], self.Rx[1]))

        while self.state == self.goal_state or self.state in self.invalid_states:
            self.state = (random.randrange(
                self.Ry[0], self.Ry[1]), random.randrange(self.Rx[0], self.Rx[1]))

        return self.state

    def step(self, action):
        self.R = self._build_rewards()
        # Evolve agent state
        state_next = (self.state[0] + self.action_coords[action][0],
                      self.state[1] + self.action_coords[action][1])
        # Collect reward
        reward = self.R[self.state + (action,)]
        # Terminate if we reach bottom-right grid corner

        done = state_next[0] < self.Ry[0] or state_next[0] > self.Ry[1] - \
            1 or state_next[1] < self.Rx[0] or state_next[1] > self.Rx[1] - \
            1 or self.state == self.goal_state
        # Update state

        self.state = state_next
        return state_next, reward, done

    def _build_rewards(self):
        R = self.r_nongoal * \
            np.ones(self.state_dim + self.action_dim, dtype=float)  # R[s,a]
        contains_goal = False

        for y in range(self.Ry[0], self.Ry[1]):
            for x in range(self.Rx[0], self.Rx[1]):
                if (y, x) == self.goal_state:
                    contains_goal = True
                if not self.on_top_edge and y == self.Ry[0] and (y - 1, x) not in self.invalid_states:
                    R[y, x, self.action_dict["up"]] = self.Z[y - 1, x]
                if not self.on_bottom_edge and y == self.Ry[1] - 1 and (y + 1, x) not in self.invalid_states:
                    R[y, x, self.action_dict["down"]] = self.Z[y + 1, x]
                if not self.on_left_edge and x == self.Rx[0] and (y, x - 1) not in self.invalid_states:
                    R[y, x, self.action_dict["left"]] = self.Z[y, x - 1]
                if not self.on_right_edge and x == self.Rx[1] - 1 and (y, x + 1) not in self.invalid_states:
                    R[y, x, self.action_dict["right"]] = self.Z[y, x+1]

        if contains_goal:
            if self.goal_state[0] - 1 > 0:
                R[self.goal_state[0] - 1, self.goal_state[1],
                    self.action_dict["down"]] = self.r_goal  # arrive from above
            if self.goal_state[0] + 1 < self.Ny:
                R[self.goal_state[0] + 1, self.goal_state[1],
                    self.action_dict["up"]] = self.r_goal  # arrive from below
            if self.goal_state[1] + 1 < self.Nx:
                R[self.goal_state[0], self.goal_state[1] + 1,
                    self.action_dict["left"]] = self.r_goal  # arrive from right
            if self.goal_state[1] - 1 > 0:
                R[self.goal_state[0], self.goal_state[1] - 1,
                    self.action_dict["right"]] = self.r_goal  # arrive from left

        return R

    def print_layout(self):
        for y in range(self.Ry[0], self.Ry[1]):
            for x in range(self.Rx[0], self.Rx[1]):
                if (y, x) in self.invalid_states:
                    print('[x]', end='')
                elif (y, x) == self.goal_state:
                    print('[G]', end='')
                else:
                    print('[ ]', end='')
            print("")

    def allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.state[0], self.state[1]

        if (y > 0) and not (y - 1, x) in self.invalid_states:  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])

        if (y < self.Ny - 1) and not (y + 1, x) in self.invalid_states:  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])

        if (x > 0) and not (y, x - 1) in self.invalid_states:  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])

        if (x < self.Nx - 1) and not (y, x + 1) in self.invalid_states:  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])

        actions_allowed = np.array(actions_allowed, dtype=int)
        #print('State: {}, allowed actions: {}'.format(self.state, actions_allowed))
        return actions_allowed
