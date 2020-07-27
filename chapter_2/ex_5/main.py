"""
Exercice 2.5: 

Design and conduct an experiment to demonstrate the 
difficulties that sample-average methods have for nonstationary problems. 
Use a modified version for the 10-armed testbed in which all the q_*(a) 
start out equal and then take independent random walks (say by adding a 
normally distributed increment with mean zero and standard deviation 0.01 
to all the q_*(a) on each step). Prepare plots like Figure 2.2. for an 
action-value method using sample averages, incrementally computed, and 
another action-value method using a constant step-size parameter, 
alpha = 0.1. Use epsilon = 0.1 and longer runs, say of 10'000 steps.
"""


from agent import ActionValueAgent
from test_bed_env import TestBedEnv

ENV_NUMBER_TEST_BEDS = 10
EPSILON = 0.1
ALPHA = 0.1
VARIANCE = 0.01 ** 2
N_STEP = 1000  # Number of time steps in the learning
N_AV = 100  # Number of runs per agents

# Setup agents and environment
agent_cst = ActionValueAgent(n_bed=ENV_NUMBER_TEST_BEDS, type_learning_rate="constant", alpha_cst=ALPHA, epsilon=EPSILON)
agent_sample_av = ActionValueAgent(n_bed=ENV_NUMBER_TEST_BEDS, type_learning_rate="sample", alpha_cst=ALPHA, epsilon=EPSILON)
env = TestBedEnv(n_bed=ENV_NUMBER_TEST_BEDS, vars=VARIANCE)

# Run simulation
agents = [agent_cst, agent_sample_av]
env.run(agents, n_step=N_STEP, n_av=N_AV)