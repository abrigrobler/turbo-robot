from trainer import Trainer
from environment import FullEnvironment
import random
from utils import Utils

# Test on a random environment
print()
print('-----------------------------------------------------------------------------------')
print()


i_s = []

goal = (28, 26)
Ny = 30
Nx = 30
'''

for i in range(Ny):
    for j in range(Nx):
        r = random.random()
        if r > 0.93 and (i, j) != goal:
            i_s.append((i, j))
'''

for i in range(30):
    if i != 7 and i != 18:
        i_s.append((i, 15))
        i_s.append((15, i))


full_env = FullEnvironment(
    goal=goal, invalid_states=i_s, Ny=Ny, Nx=Nx, r_nongoal=-0.1, r_goal=100)
region_ranges = Utils.equal_regions(Ny, Nx, 10, 10)

test_1 = Trainer(full_env, region_ranges)
test_1.train(600, update_interval=10,
             snapshot_interval=30, agent_check_interval=10)
test_1.compare_with_single_agent()
