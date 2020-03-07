import numpy as np
import random

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

    def _update_mean(self, var=1, mean=0):
        """
        Update means according to independant random walks
        """
        self.means += random.gauss(mean, var) * np.ones(self.n_bed)

    def get_rewards(self, bed_number):
        # return list of reward per action
        rew = np.random.normal(self.means, self.vars, self.n_bed)

        # update the mean according to bm
        self._update_mean()
        
        return rew

    def response(self, action):
        return 1

    def run(self, agent):
        # agent takes action
        action = agent.select_action()

        # environment responds
        rew = self.response(action)

        # agent updates his memory
        agent.update(action, rew)


if __name__ == "__main__":
    # testing unit