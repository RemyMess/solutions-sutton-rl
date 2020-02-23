import numpy as np
from random import random

class test_bed:
    """
    Defining non-stationary version of the k-armed testbed.
    Initialises all means with a similiar random value between 0 and 1, 
    evolves then according to a random walk.
    """
    def __init__(n_bed, low_b_mean=0, high_b_mean=1):
        self.means = np.ones(n_bed) * (random.random() * (high_b_mean - low_b_mean) + low_b_mean)

    def _update_mean():
        """
        Update means according to independant random walks
        """
        pass

    def get_rewards():
        # return list of reward per action

        # update the mean according to bm
