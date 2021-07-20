from trainer import Trainer
from environment import FullEnvironment

# Test on a larger environment
print()
print('-----------------------------------------------------------------------------------')
print()

i_s = []

for i in range(21):
    if i != 5 and i != 13:
        i_s.append((i, 10))
        i_s.append((10, i))


full_env = FullEnvironment(
    goal=(18, 17), invalid_states=i_s, Ny=21, Nx=21, r_nongoal=-0.1, r_goal=100)
region_ranges = [
    [(0, 13), (0, 7)],
    [(0, 13), (5, 13)],
    [(0, 13), (10, 20)],
    [(10, 20), (0, 7)],
    [(10, 20), (5, 13)],
    [(10, 20), (10, 20)],
]

test_1 = Trainer(full_env, region_ranges)
test_1.train(3000, update_interval=100)
test_1.compare_with_single_agent()
