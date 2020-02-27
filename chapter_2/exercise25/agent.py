from test_bed_env import TestBedEnv
import numpy as np
import logging

# Meta variables
ENV_NUMBER_TEST_BEDS = 100


class ActionValueAgent(TestBedEnv):
    """
    Agent working using action-value algorithm.
    """

    def __init__(self, n_it, n_bed, type_learning_rate, alpha_cst = 1):
        self.n_it = n_it
        self.env = TestBedEnv(ENV_NUMBER_TEST_BEDS)
        self.action_values = np.zeros(ENV_NUMBER_TEST_BEDS)
        self.estimate_actions = # make a dic with 2 varsw
        self.env = TestBedEnv(n_bed)

        assert(isinstance(type_learning_rate, str))
        if type_learning_rate not in ["sample", "constant"]:
            logging.error("__init__: variable 'type_learning_rate' must be either 'sample' or 'constant'.")
        else:
            if type_learning_rate == "sample":
                def alpha(self, n):
                    """
                    Learning rate alpha in the action-value update.
                    """
                    if n > 0:
                        return 1/n
                    else:
                        raise AssertionError("alpha: n must be an integer.")
            else:
                def alpha(self, n):
                    """
                    Learning rate alpha in the action-value update.
                    """
                    return alpha_cst
        
    def select_action():
        """
        Select action based on best estiaate.
        """
        # get higher action
        max = self.estimate_actions[0]
        candidates = []
        for est in self.estimate_actions:
            if est > max:
                est = max
                candidates = []
            elif est == max:
                if                


    def train(learning_rate):
        for _ in range(n_it):
            