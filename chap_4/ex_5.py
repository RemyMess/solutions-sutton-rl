import numpy as np
import matplotlib.pyplot as pyplot


REWARD_PER_CAR_SOLD = 10
REWARD_PER_CAR_MOVED_1_TO_2 = -2
REWARD_PER_CAR_MOVED_2_TO_1 = -2
LAMBDA_1_REQ = 3
LAMBDA_1_RET = 4
LAMBDA_2_REQ = 3
LAMBDA_2_RET = 2

MAX_CAR_MOVED_FROM_1_TO_2 = 3
MAX_CAR_MOVED_FROM_2_TO_1 = 3
MAX_CAR_SAME_TIME_PARK_1 = 20
MAX_CAR_SAME_TIME_PARK_2 = 20

INIT_CAR_1 = 10
INIT_CAR_2 = 10

# TRAINING HYPERPARAMETERS
GAMMA = 0.9


class policy_iteration:
    """Agent using policy iteration for car parking problem (Sutton ex 4.5)
    """
    def __init__(self, n_pol_eval_per_batch):
        self.env = car_env(LAMBDA_1_REQ,
                               LAMBDA_1_RET,
                               LAMBDA_2_REQ,
                               LAMBDA_2_RET,
                               REWARD_PER_CAR_SOLD,
                               MAX_CAR_MOVED_FROM_1_TO_2,
                               MAX_CAR_MOVED_FROM_2_TO_1,
                               MAX_CAR_SAME_TIME_PARK_1,
                               MAX_CAR_SAME_TIME_PARK_2,
                               REWARD_PER_CAR_MOVED_1_TO_2,
                               REWARD_PER_CAR_MOVED_2_TO_1)

        # data structure for states: np.array of (i,j) where
        #   - i: n_car parking 1
        #   - j: n_car parking 2
        self.states = np.array([[[i,j] for i in range(MAX_CAR_SAME_TIME_PARK_1 + 1)] for j in range(MAX_CAR_SAME_TIME_PARK_2 + 1)])
        self.actions = self.env.actions.copy()
        self.values = np.zeros(shape=(len(self.states[0]), len(self.states[1])))
        
        self.current_state = self.env.n_cars_per_park.copy()
        
        # Training hyperparams
        self.n_pol_eval_per_batch = n_pol_eval_per_batch
        
        
    def greedy_policy(self, state_x, state_y, values):
        arg_max_action = [0, 0]
        max_rew = float("-inf")
        for i,j in self.actions:
            if [i,j] != [state_x, state_y]:
                rew = self.env.rew(state_x, state_y, i, j)
                if rew > max_rew:
                    arg_max_action = [i,j]
                    max_rew = rew
                
        return arg_max_action, max_rew
    
    def arg_max_state_values(self, state_x, state_y, values):
        arg_max = 0
        max_val = float("-inf")
        for i,j in self.states:
            if [i,j] != [state_x, state_y]:
                if  values[i, j] > max_val:
                    arg_max = [i,j]
                    max_val = values[i, j]
                
        return arg_max, max_val
    
    def policy_eval(self):
        old_values = self.values.copy()
        for i, j in self.states:
            arg_max_action, arg_max_rew = self.greedy_policy(i, j, old_values)
            #
            #
            #
            # IMPLEMENT self.p(old_state,...) below in the environment
            #
            #
            #
            #
            
            rew = self.env.rew(i, j, arg_max_action[0], arg_max_action[1])
            arg_max = self.arg_max_state_values(i,j, old_values)
            self.values[state_x, state_y] = rew + GAMMA * old_values[state_x_max, state_y_max]
    
    # def policy_improvement(self, state_x, state_y, rew):
    #     old_values = self.values.copy()
    #     for i, j in self.states:
    #         state_x_max, state_y_max = self.arg_max_state_values(i, j, old_values)
    #         arg_max_action, arg_max_rew = self.greedy_policy(i, j, old_values)
    #         rew = self.env.rew(i, j, arg_max_action[0], arg_max_action[1])
    #         arg_max = self.arg_max_state_values(i,j, old_values)
    #         self.values[state_x, state_y] = rew + GAMMA * old_values[state_x_max, state_y_max]
    


INIT_CAR_1 = 10
INIT_CAR_2 = 10

class car_env:
    """ Car environment
    actions:
        n (int, values from -3 to 3; n means 'moving n car(s) from 1 to 2'
    """
    def __init__(self, lambda_1_req, lambda_1_ret, lambda_2_req, lambda_2_ret, 
                 rew_per_sold, max_car_moved_from_1_to_2, max_car_moved_from_2_to_1, max_car_same_time_park_1, max_car_same_time_park_2,
                 rew_per_car_moved_1_to_2=-2, rew_per_car_moved_2_to_1=-2,
                 init_n_car_1 = INIT_CAR_1, init_n_car_2 = INIT_CAR_2):
        self.lambda_1_req = lambda_1_req
        self.lambda_1_ret = lambda_1_ret
        self.lambda_2_req = lambda_2_req
        self.lambda_2_ret = lambda_2_ret
        
        self.max_car_moved_from_1_to_2 = max_car_moved_from_1_to_2
        self.max_car_moved_from_2_to_1 = max_car_moved_from_2_to_1
        
        self.max_car_same_time_park_1 = max_car_same_time_park_1
        self.max_car_same_time_park_2 = max_car_same_time_park_2
        
        self.rew_per_sold = rew_per_sold
        self.rew_per_car_moved_1_to_2 = rew_per_car_moved_1_to_2
        self.rew_per_car_moved_2_to_1 = rew_per_car_moved_2_to_1
        
        init_n_car_1 = init_n_car_1
        init_n_car_2 = init_n_car_2
        
        self.n_cars_per_park = np.array((init_n_car_1, init_n_car_2))
        self.actions = np.array([[[i,j] 
                                  for i in range(-self.max_car_moved_from_1_to_2, max_car_moved_from_1_to_2)] 
                                 for j in range(-self.max_car_moved_from_2_to_1, self.max_car_moved_from_2_to_1)])
        self.n_actions = max_car_same_time_park_1 * max_car_same_time_park_2 + 1

    def next_state(self, state_x, state_y, action_x, action_y):
        r = 0
        # Evening: moving cars
        if action_x + state_x <= self.max_car_same_time_park_1:
            r += self.rew_per_car_moved_1_to_2 * action_x
        else:
            r += self.rew_per_car_moved_1_to_2 * (self.max_car_same_time_park_1 - state_x)  

        if action_y + state_y <= self.max_car_same_time_park_2:
            r += self.rew_per_car_moved_2_to_1 * action_y
        else:
            r += self.rew_per_car_moved_2_to_1 * (self.max_car_same_time_park_2 - state_y)  
        
        state_x = min(max(20, state_x + action_x), 0)
        state_y = min(max(20, state_y + action_y), 0)
               
        # Day: selling and returns
        req1, req2 = self.n_car_requested()
        ret1, ret2 = self.n_car_returned()
        
        # Assume that all car returns and requests happen at the same time; 
        # i.e. more than 20 cars can simultaneously be at the shop if there are people taking them instantly after.
        if state_x + ret1 - req1 <= 20:
            car_1_to_rent = req1
            state_x = state_x + ret1 - req1
        else:
            car_1_to_rent = 20 - state_y + ret1
            state_x = 20

        if state_y + ret2 - req2 <= 20:
            car_2_to_rent = req2
            state_y = state_y + ret2 - req2
        else:
            car_2_to_rent = 20 - state_y + ret2
            state_y = 20

        r += (car_1_to_rent + car_2_to_rent) * self.rew_per_sold
        
        return state_x, state_y, r

    def p(self, old_state_x, old_state_y, action_x, action_y, new_state_x, new_state_y):
        if new_state_x > self.max_car_same_time_park_1 or new_state_y > self.max_car_same_time_park_2:
            return 0
        else:
            # Difference of 2 indep Poisson r.v. follows a Skellam distribution
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #
            return 
            
            

    def n_car_requested(self):
        # sample from a poisson with lambda_1_req and lambda_2_req
        req1 = np.random.poisson(lam=self.lambda_1_req)
        req2 = np.random.poisson(lam=self.lambda_2_req)

        return req1, req2
    
    def n_car_returned(self):
        # sample from a poisson with lambda_1_ret and lambda_2_ret
        ret1 = np.random.poisson(lam=self.lambda_1_req)
        ret2 = np.random.poisson(lam=self.lambda_1_req)

        return ret1, ret2
