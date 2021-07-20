from trainer import Trainer
from environment import FullEnvironment

# Test on a simple environment
full_env = FullEnvironment(goal=(9, 0), invalid_states=[(
    5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)], Ny=10, Nx=10, r_nongoal=-0.1)
region_ranges = [
    [(0, 3), (0, 3)],
    [(3, 9), (0, 3)],
    [(0, 9), (3, 9)]
]

test_1 = Trainer(full_env, region_ranges)
test_1.train(900, update_interval=10)
test_1.compare_with_single_agent()
