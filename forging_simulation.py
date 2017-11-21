#coding: utf-8
'''
Having specified the state-space and contingencies,
one can solve the belief updating equations (Eq. (7)) to simulate behaviour.
The (concentration) parameters of the habits were initialised to the sum of all transition probabilities: c =?uB(u).
Prior beliefs about the initial state were initialised to d = 8 for the central location for each context and zero otherwise.
Finally, prior beliefs about policies were initialised to e = 4 with the exception of the habit, where e = 0.
These concentration parameters can be regarded as the num- ber of times each state, transition or policy has been encountered in previous trials
'''

import numpy as np
import matplotlib.pyplot as plt

policy_count = 11
state_count = 8
action_count = 2
trial_count = 32

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


def simulate():


    #the behavioural and physiological responses over 32 successive trials using a format that will be adopted in subsequent figures.
    for trial_index in range(trial_count):

        # Each trial comprises two actions following an initial observation.
        initial_obserbation = [1, 0, 0, 0, 0, 0, 0] # initial position
        observations = [initial_obserbation]

        s_tau_of_each_policy_prev = [[0 for _ in range(state_count)] for i in range(policy_count)]

        for tau in range(action_count):
            max_iter = 1500

            for i in range(max_iter):
                '''
                Inference
                '''
                # state_estimation of each policy
                s_tau_of_each_policy = state_estimation(exp_ln_A, exp_ln_B, observations[-1], s_tau_of_each_policy_prev)
                # policy selecton
                exp_policy = 0
                # calc precision
                exp_beta = 0

                '''
                Learning
                '''

            # make action from policy

            # get observation


            s_tau_of_each_policy_prev = s_tau_of_each_policy
        break

    return


'''
Inference
'''
def state_estimation(exp_ln_A, exp_ln_B, current_observation, s_tau_of_each_policy_prev):

    #calc future expectations
    s_tau_future_of_each_policy = []
    s_tau_of_each_policy = []

    for pi in range(policy_count):

        s_tau_of_pi = softmax(np.dot(exp_ln_A.T, current_observation) + np.dot(exp_ln_B.T, s_tau_of_each_policy_prev[pi]) + np.dot(exp_ln_B.T, s_tau_future_of_each_policy[pi]))
        s_tau_of_each_policy.append(s_tau_of_pi)

    return s_tau_future_of_each_policy

def policy_selection():
    return

def precision():
    return

'''
Learning
'''

def update_Aa():
    return


def update_Bb():
    return


def update_Cc():
    return


def update_Dd():
    return

def update_Ee():
    return


'''
Setting and initial parameters
'''
def generate_setting():
    U = [1,2,3,4] # control state implying location
    '''
    the 11 policies considered.
    The first 10 policies correspond to staying at the center and then moving to each of the four locations,
    moving to the left or right arm and staying there, or moving to the lower arm and then moving to each of the four locations.
    The 11th policy corresponds to a habit (i.e., state-action policy).
    '''
    T = [(1,1),(1,2),(1,3),(1,4),(2,2),(3,3),(4,1),(4,2),(4,3),(4,4),('habit')] # habit or select location
    U_bf = np.array([0, 3, -3, 3, -3, 0, 0], dtype='float128').T #U_tau
    S = [(1,2),(1,3),(2,2),(2,3),(3,2),(3,3),(4,2),(4,3)] #hidden states (location, context)
    O = [i for i in range(7)] # outcomes if outcome is 5 or 6, it means get the context information
    return [U,T,U_bf,S,O]

def generate_initial_params():

    p = 0.98
    q = 1 - p

    A = np.array([
        [1,1,0,0,0,0,0,0],
        [0,0,p,q,0,0,0,0],
        [0,0,q,p,0,0,0,0],
        [0,0,0,0,q,p,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1]], dtype='float128')
    B = np.array()
    C = sum(B)
    d = np.array([8,8,0,0,0,0,0,0])
    e = np.array([0,4,4,4,4])
    return [A,B,C,d,e]

if __name__ == '__main__':
    simulate()
