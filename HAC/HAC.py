import random
import torch
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import random

import matplotlib.pyplot as plt
import pandas as pd

import copy

###
from typing import Tuple
###

##############################
class HER:

    def __init__(self, epsilon: int = 0, strategy='future'):
        """
        :param epsilon:
        :param strategy: String, expected one of ['future', 'standard']
        """
        self._epsilon = epsilon
        self._replay_buffer = []  # TODO: improve to queue
        self.strategy = strategy
        allowed_strategies = {'future', 'standard'}
        assert strategy in allowed_strategies, f"Expected one of {allowed_strategies}"

    def clear(self):
        self._replay_buffer = []

    def append(self, transition: Tuple):
        self._replay_buffer.append(transition)

    def compare_state(self, state_a, state_b):
        # TODO: add a function here based on state transitions
        return np.allclose(state_a, state_b, atol=0.1)
        # return False

    def is_in_states(self, state_a, states):
        for state_b in states:
            if self.compare_state(state_a, state_b):
                return True
        return False

    def get_reward(self, s, g):
        return 1

    def get_hindsight_goal(self, i):
        if self.strategy == 'future':
            ran = random.randint(i, len(self._replay_buffer)) -1 ###
            _, _, _, g, _, _ = self._replay_buffer[ran]
        else:  # standard
            _, _, _, g, _, _ = self._replay_buffer[-1]

        return g

    def __call__(self, all_goals):

        initial_state = self._replay_buffer[0][0]
        new_replay_buffer = []
        for i in range(len(self._replay_buffer)):
            s, a, r, s_new, goal, gamma = self._replay_buffer[i]

            hs_goal = self.get_hindsight_goal(i)

            if self.is_in_states(s_new, all_goals) or self.compare_state(s, hs_goal):
                r = self.get_reward(s_new, hs_goal)
                gamma = 0
            else:
                r = -1
                gamma = 0.9

            new_replay_buffer.append((s, a, r, s_new, hs_goal, gamma))

        self.clear()
        return new_replay_buffer
#######################



class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Policy, self).__init__()
        self.linear_1 = nn.Linear(num_inputs, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


def train_level(lvl, state , goal, subgoal_testing, total_reward):
    if episode%100 == 0:
        print("train_level lvl:" ,lvl)
        print("train_level state:" ,state)
        print("train_level goal:" ,goal)
        print("train_level subgoal_testing:" ,subgoal_testing)
    action_missed = False
    temperature = max( 0.0, (1-episode/(TEMPERATURE_TIME+1)) )

    for step in range(H):
        done = False
        action = sample(policies[lvl], state, goal, subgoal_testing, lvl, temperature=temperature)

        if lvl >0 : # if not at the bottom layer
            
            # decide whether we subgoal_test at the following lower level
            subgoal_testing_next = subgoal_testing or np.random.binomial(1,lamb,1)[0]
            
            state_next, total_reward, done = train_level(lvl-1, state, action, subgoal_testing_next, total_reward)
            
            action_np = action.detach().numpy()
            state_next_np = state_next.detach().numpy()
            action_missed = not np.allclose(state_next_np, action_np, atol=0.1)
            if action_missed:
                # reward = -1
                reward = torch.tensor(-1, dtype=torch.float)
            else:
                reward = 0
                reward = torch.tensor(0, dtype=torch.float)
                replay_buffer[lvl].append( (state, action, reward, state_next, goal, 0) ) ###
                return state_next, total_reward, done
                # done = True

#####            goals[lvl] = state
            
        else:
            # take a step in the real game
            action_np = action.detach().numpy()
            state_next_np, reward_np, done, _ = env.step(action_np)
            if episode%100 == 0:
                print('action is ', action_np)
            # turn everything into tensor immediately
            state_next = torch.tensor(state_next_np, dtype=torch.float)

            # if done:
            #     reward_np = -5

            total_reward += reward_np
            reward = torch.tensor(reward_np, dtype=torch.float)

        if lvl > 0:
            if action_missed: # action is the same as the g_i-1
                if subgoal_testing_next:
                    replay_buffer[lvl].append((state, action, -H, state_next, goal, 0))########3
                    pass
                action = state_next #######
        
        if lvl>0:
            replay_buffer[lvl].append( (state, action, reward, state_next, goal, gamma if action_missed else 0) ) ###
        else:
            replay_buffer[lvl].append( (state, action, reward, state_next, goal, 0 if done else gamma) ) ###

        if USE_HER:
            her_storage[lvl].append( (state, action, None, state_next, None, None ) )

# # ######### This section of code is for debugging:
#         if episode%200 == 0 and lvl==0:
#             # state_goal = np.concatenate((state, goal))
#             # policy_input = torch.from_numpy(np.expand_dims(state_goal, axis=0)).float()
#             policy_input = torch.cat((state, goal))

#             # state_action_goal = np.concatenate((state, sample(policies[lvl], state_next, goal, False, lvl).reshape((-1,)), goal))
#             # Q_input = torch.from_numpy(np.expand_dims(state_action_goal, axis=0)).float()
#             Q_input = torch.cat((state, action.float().reshape((-1,)), goal))

#             test_pol = policies[lvl].forward(policy_input).detach().numpy()
#             test_Q = Q_networks[lvl](Q_input)

#             print(' ')
#             print('lvl: ', lvl, 'step: ', step)
#             print('pol', test_pol)
#             print('Q', test_Q)
#             print('action taken', action)
#             print('## state_next', state_next)
#             print(' ')

# # ######### /end section of code for debugging

        state = state_next

        if len(replay_buffer[lvl]) > 10000:
            replay_buffer[lvl] = replay_buffer[lvl][-10000:]
        
        if done:
            break
            # return state_next, total_reward, done #########used to be break
    
    if USE_HER:
        replay_buffer[lvl].extend(her_storage[lvl](all_goals=[]))
    return state_next, total_reward, done


def sample(policy, state, goal, subgoal_testing, lvl, temperature=0.0):
    state_goal = torch.cat((state, goal))
    policy_input = state_goal

    if lvl==0:
        if subgoal_testing:
            action = torch.argmax(policy.forward(policy_input))
        else:
            if random.random()<temperature:
                action = torch.randint(0,2, (1,))[0]
            else:
                weights = F.softmax(policy.forward(policy_input))
                action = torch.multinomial(weights, 1)[0]
    else:
        if subgoal_testing:
            action = policy.forward(policy_input)
        else:
            action = policy.forward(policy_input)
            action = action + torch.randn(action.shape)

        # action = top_goal
    return action


USE_HER = False
TEMPERATURE_TIME = 1000 # set to 0 if you want


num_episodes = 5000
num_levels = 3
top_goal = torch.tensor([0, 0, 0, 0], dtype=torch.float) # env-specific
H = 15 #40
lamb = 0.1

learning_rate = 0.005

replay_buffer = [ [] for lvl in range(num_levels) ]
her_storage = [ [] for lvl in range(num_levels) ]
her_storage = [ HER() for lvl in range(num_levels) ]

gamma = 0.9

policies =[]
Q_networks = []
p_optimizers = []
q_optimizers = []

policies_prime=[]
Q_networks_prime=[]

# initialising policy networks and Q networks
for i in range(num_levels):
    if i==0:
        pol = Policy(8, 2, 10)
        q_n = Policy(4+1+4, 1, 10) # we use the same network class but with different shape
        pol_prime = Policy(8, 2, 10)###
        q_n_prime = Policy(4+1+4, 1, 10) # we use the same network class but with different shape###
    else:
        pol = Policy(8, 4, 10) # input dim is state_dim + goal_dim
        q_n = Policy(4+4+4, 1, 10) # input dim is state_dim + action_dim + goal_dim
        pol_prime = Policy(8, 4, 10) # input dim is state_dim + goal_dim###
        q_n_prime = Policy(4+4+4, 1, 10) # input dim is state_dim + action_dim + goal_dim###
    policies.append(pol)
    Q_networks.append(q_n)
    policies_prime.append(pol_prime)###
    Q_networks_prime.append(q_n_prime)###
    p_optimizers.append(optim.Adam(pol.parameters(), lr=learning_rate))
    q_optimizers.append(optim.Adam(q_n.parameters(), lr=learning_rate))

# policies_prime = copy.deepcopy(policies)
# Q_networks_prime = copy.deepcopy(Q_networks)


def update_parameters():
    tau = 0.01 # hyperparameter update rate for target network 0 <   1
    N = 8# hyperparameter number of instances we sample

    for lvl in range(num_levels): # iterate through all levels
        if len(replay_buffer[lvl]) <= N:
            break
        # sample N instances from the replay_buffer[i]
        training_batch = random.sample(replay_buffer[lvl], N)
        y = []
        Qi_vec = []

        Q_prime_lvl = Q_networks_prime[lvl]
        pi_prime_lvl = policies_prime[lvl]

        Q_lvl = Q_networks[lvl]
        pi_lvl = policies[lvl]

        Q_opt_lvl = q_optimizers[lvl]
        p_opt_lvl = p_optimizers[lvl]

        Lcritic = torch.tensor([0.0], requires_grad=True, dtype=torch.float) ####
        for state, action, reward, next_state, goal, gamma in training_batch:

            state_action_goal = torch.cat((state, action.float().reshape((-1,)), goal))

            Qi_vec.append( Variable(Q_lvl(state_action_goal), requires_grad=True) )
            Q_val = Q_lvl(state_action_goal)
            Q_val.retain_grad()
            Qi_vec.append( Q_val)

            act = sample(pi_prime_lvl, next_state, goal, False, lvl)
            state_action_goal = torch.cat((next_state, act.float().reshape((-1,)), goal))

            yn = reward + torch.tensor(gamma, dtype=torch.float)*Q_prime_lvl(state_action_goal).detach()
            if episode%100 == 0 and lvl==0:
                print('in update_parameters, gamma= ', gamma)
                print('in update_parameters, yn= ', yn)
                print('in update_parameters, Q= ', Q_lvl(state_action_goal))
                print('in update_parameters, Q.parameters= ', list(Q_lvl.parameters())[0][0])

            Lcritic = Lcritic + (Q_val - yn)**2 ###

            y.append(yn)

        Q_opt_lvl.zero_grad()
        # critic_loss = nn.MSELoss()
        # Lcritic = critic_loss( torch.tensor(Qi_vec, requires_grad=True), torch.tensor(y) )
        Lcritic.backward(retain_graph=True)
        Q_opt_lvl.step()
        if episode%100 == 0 and lvl==0:
            print('in update_parameters, Lcritic= ', Lcritic)
            print('Lcritic grads ', list(Q_lvl.parameters())[0].grad)


        Lactor = torch.tensor([0.0], requires_grad=True, dtype=torch.float)
        if lvl > 0: # not bottom level
            i = 0
            for state, action, reward, next_state, goal, gamma in training_batch:
                diff = torch.tensor( sample(pi_lvl, state, goal, False, lvl) - action, requires_grad=True, dtype=torch.float)
                Lactor = Lactor - torch.exp(-diff.dot(diff)) * Qi_vec[i]
                i +=1
        else: # bottom level lvl = 0
            for state, action, reward, next_state, goal, gamma in training_batch:

                state_goal = torch.cat((state, goal))

                logits = pi_lvl(state_goal)
                probs = F.softmax(logits).squeeze()

                weights = []
                for act in range(2): #all_actions # '2' is env-specific
                    ac = torch.tensor(act, dtype=torch.float)
                    state_action_goal = torch.cat((state, ac.float().reshape((-1,)), goal))

                    Q_a = weights.append( Q_prime_lvl(state_action_goal) ) ############

                Lactor = Lactor - torch.dot( probs, torch.tensor(weights, requires_grad=True) ) 

        p_opt_lvl.zero_grad()
        Lactor.backward(retain_graph=True)
        # if episode%100 == 0 and lvl==0:
        #     print('in update_parameters, Lactor= ', Lactor)
        #     print('Lactor grads ', list(pi_lvl.parameters())[0].grad)
        p_opt_lvl.step()


rewards = []
# main loop
for episode in range(num_episodes):

    if episode % 10 == 0:
        # policies_prime = copy.deepcopy(policies)
        # Q_networks_prime = copy.deepcopy(Q_networks)
        for i in range(len(policies)):
            policies_prime[i].load_state_dict(policies[i].state_dict())
            Q_networks_prime[i].load_state_dict(Q_networks[i].state_dict())


    env = gym.make('CartPole-v0')
    state_np = env.reset()
    state = torch.tensor(state_np, dtype=torch.float)

    _, total_reward, _ = train_level(2, state, top_goal, False, 0)
    print('ep ',episode,', total_reward: ', total_reward)
    rewards.append(total_reward)

    update_parameters() 

env.close()

smoothed_rewards = pd.Series.rolling(pd.Series(rewards), 10).mean()
smoother_rewards = pd.Series.rolling(pd.Series(rewards), 100).mean()
smoothed_rewards = [elem for elem in smoothed_rewards]

plt.plot(rewards)
plt.plot(smoothed_rewards)
plt.plot(smoother_rewards)
plt.plot()
plt.show()


