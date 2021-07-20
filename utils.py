# Preliminary: imports
import numpy as np


class Utils:

    def build_Q(q_list):
        d = q_list.shape
        Q_augmented = np.zeros((d[1], d[2], d[3]), dtype=float)
        for i in range(d[1]):
            for j in range(d[2]):
                for k in range(d[3]):
                    Q_augmented[i, j, k] = np.max(q_list[:, i, j, k])
        return Q_augmented

    def build_greedy_policy(Q):
        d = Q.shape

        greedy_policy = np.zeros((d[0], d[1]), dtype=int)
        for y in range(d[0]):
            for x in range(d[1]):
                greedy_policy[y, x] = np.argmax(Q[y, x, :])
        return greedy_policy

    def display_greedy_policy(Q, invalid_states):
        d = Q.shape

        greedy_policy = np.zeros((d[0], d[1]), dtype=str)

        for y in range(d[0]):
            for x in range(d[1]):
                if (y, x) in invalid_states:
                    greedy_policy[y, x] = "█"
                else:
                    action = np.argmax(Q[y, x, :])
                    if action == 0:
                        greedy_policy[y, x] = "^"
                    elif action == 1:
                        greedy_policy[y, x] = ">"
                    elif action == 2:
                        greedy_policy[y, x] = "V"
                    elif action == 3:
                        greedy_policy[y, x] = "<"
        print("\nFull greedy policy(y, x):")
        print(greedy_policy)
        print()

    def calculate_average_episode_length(episodes):
        avg = []
        for n in range(len(episodes[0])):
            t = 0
            for i in range(len(episodes)):
                t += episodes[i][n]
            t = float(t)/float(len(episodes))
            avg.append(t)
        return avg

    def equal_regions(Ny, Nx, rows, columns):
        ranges_y = []
        ranges_x = []

        for i in range(Ny):
            if (i % int(Ny/rows)) == 0 and i != 0 and i < Ny:
                ranges_y.append((i - int(Ny/rows), i))
        ranges_y.append((Ny - int(Ny/rows), Ny - 1))

        for i in range(Nx):
            if (i % int(Nx/columns)) == 0 and i != 0 and i < Nx:
                ranges_x.append((i - int(Nx/columns), i))
        ranges_x.append((Nx - int(Nx/columns), Nx - 1))

        regions = []
        for y in ranges_y:
            for x in ranges_x:
                regions.append([y, x])

        return regions
