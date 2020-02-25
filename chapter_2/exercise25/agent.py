from test_bed_env import TestBedEnv
import numpy as np
import logging

# Meta variables
ENV_NUMBER_TEST_BEDS = 100


class ActionValueAgent(n_it, type_learning_rate, alpha_cst=1):
    """
    Agent working using action-value algorithm.
    """

    def __init__(self, n_it):
        self.n_it = n_it
        self.env = TestBedEnv(ENV_NUMBER_TEST_BEDS)
        self.action_values = np.zeros(ENV_NUMBER_TEST_BEDS)

        assert(isinstance(type_learning_rate, str))
        if type_learning_rate not in ["sample", "constant"]:
            logging.error("__init__: variable 'type_learning_rate' must be either 'sample' or 'constant'.")
        else:
            if "sample":
            def alpha(self, n):
                """
                Learning rate alpha in the action-value update.
                """
        

    def train(learning_rate):
        if learning_rate


        for _ in range(n_it):            
            on