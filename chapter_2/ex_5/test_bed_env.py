import numpy as np
import random
import matplotlib.pyplot as plt
import copy


# Meta variables
random.seed = 1


class TestBedEnv:
    """
    Defining non-stationary version of the k-armed testbed.
    Initialises all means with a similiar random value between 0 and 1, they then
    evolves according to a random walk.
    """
    def __init__(self, n_bed, low_b_mean=0, high_b_mean=1, vars=1):
        self.n_bed = n_bed
        self.means = np.ones(n_bed) * (random.random() * (high_b_mean - low_b_mean) + low_b_mean)
        self.vars = np.ones(n_bed) * vars

    def _update_env(self, var=1, mean=0):
        """
        Update means according to independant random walks
        """
        self.means += np.random.normal(mean, var, self.n_bed)

    def optimal_actions(self):
        """
        Return a list of the best actions.
        """
        max_mean = np.max(self.means)
        candidates = []
        for i in range(self.n_bed):
            if self.means[i] == max_mean:
                candidates.append(i)
                # print(self.means, max_mean, candidates)

        return candidates

    def track_optimal_action(self, agent, suggested_action):
        if suggested_action in self.optimal_actions():
            agent.optimal_action_number_tracker += 1

        agent.optimal_action_ratio_tracker = np.append(agent.optimal_action_ratio_tracker, agent.optimal_action_number_tracker / agent.n_it)

    def get_rewards(self, action):
        # return list of reward per action
        return np.random.normal(self.means[action], self.vars[action])

    def response(self, action):
        # get reward
        rew = self.get_rewards(action)

        # update environment
        self._update_env()

        return rew

    def run(self, agents: list, n_step = 10000, n_av = 100, plot=True):
        averages = {}
        percentages = {}
        for i in range(1, n_av + 1):
            copy_agents = copy.deepcopy(agents)
            for agent in copy_agents:
                for _ in range(n_step):
                    # agent takes action
                    action = agent.select_action()

                    # track optimality
                    self.track_optimal_action(agent, action)

                    # environment responds
                    rew = self.response(action)

                    # agent updates his memory
                    agent.update(action, rew)

                if agent.type_learning_rate not in averages or agent.type_learning_rate not in percentages:
                    averages[agent.type_learning_rate] = agent.rewards_tracker
                    percentages[agent.type_learning_rate] = agent.optimal_action_ratio_tracker
                else:
                    averages[agent.type_learning_rate] *= (i-1)
                    averages[agent.type_learning_rate] += agent.rewards_tracker
                    
                    averages[agent.type_learning_rate] *= 1/i
                    
                    percentages[agent.type_learning_rate] *= (i-1)
                    percentages[agent.type_learning_rate] += agent.optimal_action_ratio_tracker
                    percentages[agent.type_learning_rate] /= i

        print(f"Results of averages: {averages}")                
        print(f"Results of percentages: {percentages}")                
        
        if plot:
            self.plot_average_rewards(averages)
            self.plot_optimality_actions(percentages)
            
        
    def plot_average_rewards(self, averages: dict):
        for type_learning_rate in averages:
            plt.plot(averages[type_learning_rate], label=type_learning_rate)
            
        plt.title(f"Agent with epsilon=0.1 and alpha=0.1")
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.legend()
        
        # plt.savefig("average_reward.png",)
        plt.show()

    def plot_optimality_actions(self, percentages: dict):
        for type_learning_rate in percentages:
            plt.plot(percentages[type_learning_rate], label=type_learning_rate)
            
        plt.title(f"Agent with epsilon=0.1 and alpha=0.1")
        plt.xlabel("Steps")
        plt.ylabel("% Optimal action")
        plt.legend()
        
        plt.show()