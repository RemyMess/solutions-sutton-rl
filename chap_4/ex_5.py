import numpy as np
import matplotlib.pyplot as pyplot
import scipy
import math
import logging

REWARD_PER_CAR_SOLD = 10
REWARD_PER_CAR_MOVED_1_TO_2 = -2
REWARD_PER_CAR_MOVED_2_TO_1 = -2
LAMBDA_1_REQ = 3
LAMBDA_1_RET = 4
LAMBDA_2_REQ = 3
LAMBDA_2_RET = 2

MAX_CAR_MOVED_FROM_1_TO_2 = 2
MAX_CAR_MOVED_FROM_2_TO_1 = 2
MAX_CAR_SAME_TIME_PARK_1 = 10
MAX_CAR_SAME_TIME_PARK_2 = 10

INIT_CAR_1 = 3
INIT_CAR_2 = 3

MAX_MONEY_EARNED_LOST = 120

# TRAINING HYPERPARAMETERS
GAMMA = 0.9
POLICY_EVAL_DELTA = 0.1
POLICY_ITER_N = 100

class policy_iteration:
    """Agent using policy iteration for car parking problem (Sutton ex 4.5)
    """
    def __init__(self):
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
                               REWARD_PER_CAR_MOVED_2_TO_1,
                               GAMMA)

        # data structure for states: np.array of (i,j) where
        #   - i: n_car parking 1
        #   - j: n_car parking 2
        self.states = np.array([[[i,j] for i in range(MAX_CAR_SAME_TIME_PARK_1 + 1)] for j in range(MAX_CAR_SAME_TIME_PARK_2 + 1)])
        self.states = self.states.reshape(1, -1, 2).squeeze()
        self.actions = self.env.actions.copy()
        self.values = np.random.rand(MAX_CAR_SAME_TIME_PARK_1 + 1, MAX_CAR_SAME_TIME_PARK_2 + 1)
        self.current_state = self.env.n_cars_per_park.copy()
        
        # Training hyperparams
        self.n_pol_eval_per_batch = POLICY_ITER_N
        self.max_money = MAX_MONEY_EARNED_LOST 
        self.gamma = GAMMA
        
    def greedy_policy(self, state_x, state_y, values):
        max_val = float("-inf")
        for action in self.env.actions:
            new_val = 0
            for rew in range(-self.max_money, self.max_money):
                for new_state_x, new_state_y in self.states:
                    new_val += self.env.p(new_state_x, new_state_y, rew, state_x, state_y, action) * (rew + GAMMA * values[new_state_x, new_state_y])
            if new_val > max_val:
                max_val = new_val
                arg_max = action 
                
        return arg_max, max_val
    
    def policy_eval(self):
        old_values = self.values.copy()
        delta = POLICY_EVAL_DELTA
        while delta >= POLICY_EVAL_DELTA:
            delta = POLICY_EVAL_DELTA
            logging.warning(f"New round of policy evaluation: delta: {delta}")
            for state_x, state_y in self.states:
                action = self.greedy_policy(state_x, state_y, old_values)[0]
                new_val = 0
                for new_state_x, new_state_y in self.states:
                    for rew in range(-MAX_MONEY_EARNED_LOST, MAX_MONEY_EARNED_LOST):
                        p = self.env.p(rew, new_state_x, new_state_y, state_x, state_y, action)
                        new_val += p * (rew + old_values[state_x, state_y])
                        if p != 0:
                            logging.warning(f"new_states: ({new_state_x}, {new_state_y}), rew: {rew}, p: {p}; action: {action}" )
                    delta = min(abs(new_val - old_values[state_x, state_y]), delta)
                self.values[state_x, state_y] = new_val
                logging.warning(f"policy evaluation: states: ({state_x}, {state_y}); action: {action}; old value: {old_values[state_x, state_y]}; new value: {new_val}; delta: {new_val - old_values[state_x, state_y]}")
    
    def policy_iter(self):
        for _ in range(POLICY_ITER_N):
            self.policy_eval()
        logging.warning("Done")
        logging.warning(f"Values: {self.values}")


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
        # action (int): action = 2 means: "2 cars are moved from 1 to 2"; action=-2 means "2 cars moved from 2 to 1"
        self.actions = np.array(range(-max_car_moved_from_2_to_1, max_car_moved_from_1_to_2 + 1))

    def next_state(self, state_x, state_y, action):
        r = 0
        # Evening: moving cars
        if state_x - action <= self.max_car_same_time_park_1:
            r += self.rew_per_car_moved_1_to_2 * abs(action)
        else:
            r += self.rew_per_car_moved_1_to_2 * (self.max_car_same_time_park_1 - state_x)  

        if state_y + action <= self.max_car_same_time_park_2:
            r += self.rew_per_car_moved_2_to_1 * abs(action)
        else:
            r += self.rew_per_car_moved_2_to_1 * (self.max_car_same_time_park_2 - state_y)  
        
        state_x = min(max(20, state_x - action), 0)
        state_y = min(max(20, state_y + action), 0)
               
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

    def p(self, new_state_x, new_state_y, reward, old_state_x, old_state_y, action):
        if new_state_x > self.max_car_same_time_park_1 or new_state_y > self.max_car_same_time_park_2 or old_state_x > self.max_car_same_time_park_1 or old_state_y > self.max_car_same_time_park_2:
            return 0
        else:
            # Evening operations
            morning_car_1 = old_state_x - action
            morning_car_2 = old_state_y + action
            
            reward_from_op = 0
            if action > 0:
                reward_from_op = action * self.rew_per_car_moved_1_to_2
            else:
                reward_from_op = -action * self.rew_per_car_moved_2_to_1
            
            # Daily operation
            if reward + reward_from_op < 0:
                return 0
            else:
                n_car_to_rent = (reward + reward_from_op) // self.rew_per_sold

                # Probability of event: 
                #   P({X_1 + Y_1 = n_car_to_rent}
                #       ^ {X_1 <= morning_car_1 + Y_1}
                #       ^ {X_2 <= morning_car_2 + Y_2}
                #       ^ {new_state_x = old_state_x + Y_1 - X_1}
                #       ^ {new_state_y = old_state_y + Y_2 - X_2})
                #   = P( {X_1 + X_2 = n_car_to_rent}) 
                #       * P({new_state_x - old_state_x + new_state_y - old_state_y + n_car_to_rent = Y_1 + Y_2})
                #       * 1_{new_state_x - old_state_x >= - morning_car_1}
                #       * 1_{new_state_y - old_state_y >= - morning_car_2}
                # 
                # with  X_1, X_2 = cars to rent in location 1, 2
                #       Y_1, Y_2 = returned cars in location 1, 2
                # 
                # NOTE: sum of Poisson is Poisson with parameter equal to sum of params
                if new_state_x - old_state_x < - morning_car_1 or new_state_y - old_state_y < - morning_car_2:
                    return 0
                elif new_state_x - old_state_x + new_state_y - old_state_y + n_car_to_rent < 0:
                    return 0
                else:
                    def poisson_pdf(lamb, k):
                        return lamb**k * math.exp(-lamb) / (math.factorial(k))

                    # def poisson_cdf(lamb, k):
                    #     return math.exp(-lamb) * np.sum([lamb**i/math.factorial(i) for i in range(math.floor(k) + 1)])
            
                    prob = poisson_pdf(self.lambda_1_req + self.lambda_2_req, n_car_to_rent) * poisson_pdf(self.lambda_1_req + self.lambda_2_req, new_state_x - old_state_x + new_state_y - old_state_y + n_car_to_rent)
                    
                    return prob
            
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


if __name__ == "__main__":
    obj = policy_iteration()
    obj.policy_iter()
    # obj.env.p(6,7,-77,4,5,2)
