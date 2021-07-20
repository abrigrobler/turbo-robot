#Preliminary: imports
import numpy as np
import time
import matplotlib.pyplot as plt
from console_progressbar import ProgressBar
from utils import Utils
from agent import Agent, Player


class Trainer():
    def __init__(self, full_env, region_ranges):
        np.set_printoptions(edgeitems=30, linewidth=100000,
                            formatter=dict(float=lambda x: "%.3g" % x))
        self.full_env = full_env
        self.region_ranges = region_ranges
        self.baseline_agent = Agent(full_env)
        self.regional_agents = []
        self.baseline_policies = []
        self.regional_policies = []
        print('Environment to be tested:')
        # full_env.print_layout()

    def train(self, num_episodes, update_interval=100, snapshot_interval=100, agent_check_interval=20):

        # Train single agent
        print('Training single agent...')
        pb = ProgressBar(total=num_episodes)
        t_time = time.time()
        for i in range(num_episodes):
            self.baseline_agent.run_single_episode()
            if i % snapshot_interval == 0:
                self.baseline_policies.append(
                    Utils.build_greedy_policy(self.baseline_agent.Q))
            #pb.print_progress_bar(i + 1)
        print('Training time for single agent was {}s'.format(time.time() - t_time))

        print('Populating sub environments...')
        self.full_env.set_sub_envs(self.region_ranges)

        print('Training regional agents...')

        for se in self.full_env.sub_envs:
            self.regional_agents.append(Agent(se))

        t_time = time.time()
        for i in range(num_episodes):
            episode_q_list = []

            for a in self.regional_agents:
                a.run_single_episode()
                episode_q_list.append(a.Q)

            '''
            if i % agent_check_interval == 0:  # Stop training unnecessary agents
                for a in self.regional_agents:
                    if a.last_episode_reward <= 0 and a.training:
                        a.training = False
            '''

            if i % update_interval == 0:  # Update the goal distribution function every k episodes

                self.full_env.update_z(np.array(episode_q_list))
                #print('Agents active in last interval {}'.format(training_agents))

            if i % snapshot_interval == 0:
                self.regional_policies.append(Utils.build_greedy_policy(
                    Utils.build_Q(np.array(episode_q_list))))
            # pb.print_progress_bar(i+1)

        print('Training time on regions was {}s'.format(time.time() - t_time))

        # print('Single agent policy')
        # Utils.display_greedy_policy(
        # self.baseline_agent.Q, self.full_env.invalid_states)
        # print('Regional agents policy')
        q_list = []
        for a in self.regional_agents:
            q_list.append(a.Q)
        full_Q = Utils.build_Q(np.array(q_list))
        #Utils.display_greedy_policy(full_Q, self.full_env.invalid_states)

    def compare_with_single_agent(self, num_episodes=100):
        print('Testing policy snapshots')

        baseline_avg_reward = []
        regional_avg_reward = []

        pb = ProgressBar(total=len(self.baseline_policies))

        for p in range(len(self.baseline_policies)):
            # pb.print_progress_bar(p+1)
            baseline_nav = Player(
                self.full_env, self.baseline_policies[p], num_episodes)
            baseline_avg_reward.append(baseline_nav.test_policy())
            regional_nav = Player(
                self.full_env, self.regional_policies[p], num_episodes)
            regional_avg_reward.append(regional_nav.test_policy())

        print("Plotting results...")

        # Plot data
        plt.title('Average return as training continues')
        plt.xlabel('Snapshot number')
        plt.ylabel('Average Return')
        baseline, = plt.plot(baseline_avg_reward, label='Single agent')
        subdivided, = plt.plot(regional_avg_reward, label='Multiple agents')
        plt.legend(handles=[baseline, subdivided])
        plt.show()
