#Preliminary: imports
import numpy as np
import random
import time
from console_progressbar import ProgressBar


class Agent:

    def __init__(self, env, showProgressBar=True):
        # Store state and action dimension
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # Agent learning parameters
        self.epsilon = 1.0  # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta = 0.99  # learning rate
        self.gamma = 0.99  # reward discount factor
        # Initialize Q[s,a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)
        # To suppress output if necessary
        self.showProgressBar = showProgressBar
        self.training_time = 0
        self.episode_length = []
        self.training = True
        self.last_episode_reward = 0

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(
                Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def update_Q(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * \
            (reward + self.gamma*np.max(self.Q[state_next]) - self.Q[sa])

    def run_single_episode(self):

        # Generate an episode
        iter_episode, reward_episode = 0, 0
        state = self.env.generate_start()  # starting state
        Q_prev = self.Q.copy()
        while True:
            action = self.get_action(self.env)  # get action
            state_next, reward, done = self.env.step(
                action)  # evolve state by action
            self.update_Q((state, action, state_next,
                           reward, done))  # train agent
            iter_episode += 1
            reward_episode += reward
            if done:
                break
            state = state_next  # transition to next state

        # Decay agent exploration parameter
        self.episode_length.append(iter_episode)
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.001)
        self.last_episode_reward = reward_episode
        '''
        # Calculate MSE to allow early stopping:
        Q_diff = np.subtract(self.Q, Q_prev)
        Q_diff_squared = np.square(Q_diff)
        mse = Q_diff_squared.mean()

        if mse <= 1e-10 and self.last_episode_reward > 0:
            self.training = False
        '''

    def train(self, num_episodes):
        self.training_time = time.time()
        self.episode_length = []
        pb = ProgressBar(total=num_episodes - 1)
        for episode in range(num_episodes):

            # Generate an episode
            iter_episode, reward_episode = 0, 0
            state = self.env.generate_start()  # starting state
            k = 0
            while True:
                k += 1
                action = self.get_action(self.env)  # get action
                state_next, reward, done = self.env.step(
                    action)  # evolve state by action
                self.update_Q((state, action, state_next,
                               reward, done))  # train agent
                iter_episode += 1
                reward_episode += reward
                if done:
                    break
                state = state_next  # transition to next state
            if self.showProgressBar:
                pb.print_progress_bar(episode)

            # Decay agent exploration parameter
            self.episode_length.append(k)
            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.001)

            # Print
            """
            if (episode == 0) or (episode + 1) % 10 == 0:
                print("[episode {}/{}] eps = {:.3F} -> iter = {}, rew = {:.1F}".format(
                    episode + 1, num_episodes, self.epsilon, iter_episode, reward_episode))
            """

            # Print greedy policy
            if (episode == num_episodes - 1):
                self.training_time = time.time() - self.training_time
                print('Trained in {} s'.format(self.training_time))
                self.display_greedy_policy()
                # self.display_value_function()
                """
                for (key, val) in sorted(self.env.action_dict.items(), key=operator.itemgetter(1)):
                    print(" action['{}'] = {}".format(key, val))
                print()
                """

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros(
            (self.state_dim[0], self.state_dim[1]), dtype=str)
        for y in range(self.state_dim[0]):
            for x in range(self.state_dim[1]):
                if (y, x) in self.env.invalid_states:
                    greedy_policy[y, x] = "x"
                else:
                    action = np.argmax(self.Q[y, x, :])
                    if action == 0:
                        greedy_policy[y, x] = "^"
                    elif action == 1:
                        greedy_policy[y, x] = ">"
                    elif action == 2:
                        greedy_policy[y, x] = "V"
                    elif action == 3:
                        greedy_policy[y, x] = "<"
        print("\nGreedy policy(y, x):")
        print(greedy_policy)
        print()

    def display_value_function(self):
        V = np.zeros(
            (self.state_dim[0], self.state_dim[1]), dtype=float)
        for x in range(self.state_dim[0]):
            for y in range(self.state_dim[1]):
                V[y, x] = int(np.max(self.Q[y, x, :]))
        print("\nValue function (y, x)")
        print(V)
        print()


class Player():
    def __init__(self, env, policy, num_episodes):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes

    def get_action(self):
        state = self.env.state
        action = self.policy[state[0], state[1]]
        allowed_actions = self.env.allowed_actions()
        if action not in allowed_actions:
            action = np.random.choice(allowed_actions)
        return action

    def test_policy(self):
        rewards = []
        for n in range(self.num_episodes):
            # Generate an episode
            iter_episode, reward_episode = 0, 0
            state = self.env.generate_start()  # starting state
            while True:
                action = self.get_action()
                state_next, reward, done = self.env.step(
                    action)  # evolve state by action

                iter_episode += 1
                reward_episode += reward
                if done or iter_episode >= 1000:  # Set an episode length limit
                    break

                state = state_next  # transition to next state

            rewards.append(reward_episode)
        return np.mean(rewards)
