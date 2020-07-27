import numpy as np
import logging
import random

class ActionValueAgent:
    """
    Agent working using action-value algorithm.
    """

    def __init__(self, n_bed, type_learning_rate, epsilon = 0.1, alpha_cst = 1):
        self.n_bed = n_bed
        self.action_values = np.zeros(n_bed)
        self.estimate_actions = np.zeros(n_bed)
        self.action_estimation_init = list(range(n_bed))
        self.n_it = 1
        self.alpha_cst = alpha_cst
        self.type_learning_rate = type_learning_rate
        self.epsilon = epsilon

        self.optimal_action_number_tracker = 0
        self.optimal_action_ratio_tracker = np.array([])
        
        self.rewards_tracker = np.array([])
        

        assert(isinstance(type_learning_rate, str))
        if type_learning_rate not in ["sample", "constant"]:
            logging.error("__init__: variable 'type_learning_rate' must be either 'sample' or 'constant'.")
        else:
            if type_learning_rate == "sample":
                def alpha():
                    """
                    Learning rate alpha in the action-value update.
                    """
                    return 1 / self.n_it
            else:
                def alpha():
                    """
                    Learning rate alpha in the action-value update.
                    """
                    return alpha_cst

            setattr(self, "alpha", alpha)
        
    def alpha(self):
        raise NotImplementedError()

    def select_action(self):
        """
        Select action based on best estimate with probaility 1-epsilon.
        """
        # check if all values have been tried.
        if len(self.action_estimation_init) == 0:
            # epsilon test for randomly selected action
            _will_select_random = random.choices([True, False], (self.epsilon, 1 - self.epsilon))[0]

            if _will_select_random:
                return random.choices(list(range(len(self.estimate_actions))))
            else:
                max_rew = self.estimate_actions[0]
                candidates = [0]
                for i in range(1, len(self.estimate_actions)):
                    if self.estimate_actions[i] < max_rew:
                        continue
                    elif self.estimate_actions[i] == max_rew:
                        candidates.append(i)
                    elif self.estimate_actions[i] > max_rew:
                        max_rew = self.estimate_actions[i]
                        candidates = [i]
                    
                best = random.choice(candidates)
                self.n_it += 1
                return best

        else:
            pick = random.choice(list(range(len(self.action_estimation_init))))
            action = self.action_estimation_init[pick]
            del self.action_estimation_init[pick]
            self.n_it += 1
            return action
    
    def update(self, action, reward):
        self.estimate_actions[action] = reward + self.alpha() * (self.estimate_actions[action] - reward)
        
        # keep track of average rewards over all the beds
        self.rewards_tracker = np.append(self.rewards_tracker, reward)
