import numpy as np
import random

# Meta variables
random.seed = 1
ENV_NUMBER_TEST_BEDS = 100

class TestBedEnv:
    """
    Defining non-stationary version of the k-armed testbed.
    Initialises all means with a similiar random value between 0 and 1, they then
    evolves according to a random walk.
    """
    def __init__(self, n_bed=ENV_NUMBER_TEST_BEDS, low_b_mean=0, high_b_mean=1, vars=1):
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
                print(self.means, max_mean, candidates)

        return candidates

    def track_optimal_action(self, agent, suggested_action):
        if suggested_action in self.optimal_actions():
            agent.optimal_action_number_tracker += 1

        agent.optimal_action_ratio_tracker.append(agent.optimal_action_number_tracker / agent.n_it)

    def get_rewards(self, action):
        # return list of reward per action
        return np.random.normal(self.means[action], self.vars[action])

    def response(self, action):
        # get reward
        rew = self.get_rewards(action)

        # update environment
        self._update_env()

        return rew

    def run(self, agent, n_it = 10000):
        for _ in range(n_it):
            # agent takes action
            action = agent.select_action()

            # track optimality
            self.track_optimal_action(agent, action)

            # environment responds
            rew = self.response(action)

            # agent updates his memory
            agent.update(action, rew)

        agent.plot_optimality_actions()

if __name__ == "__main__":
    # Init
    from agent import ActionValueAgent
    ENV_NUMBER_TEST_BEDS = 10
    n_it = 1000

    agent_cst = ActionValueAgent(n_bed=ENV_NUMBER_TEST_BEDS, type_learning_rate="constant", alpha_cst=0.05)
    agent_sample_av = ActionValueAgent(n_bed=ENV_NUMBER_TEST_BEDS, type_learning_rate="sample", alpha_cst=0.05)
    
    env = TestBedEnv(n_bed=ENV_NUMBER_TEST_BEDS)

    # Run simulation
    for agent in [agent_cst, agent_sample_av]:
        env.run(agent_cst)