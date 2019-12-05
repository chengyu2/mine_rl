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

###
import random
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
        return np.allclose(state_a, state_b, atol=1.0)
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
                gamma = gamma

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
        # x = F.softmax(x, dim=1)
        return x


def train_level(lvl, state , goal, subgoal_testing, total_reward):
    if episode%100 == 0:
        print("train_level lvl:" ,lvl)
        print("train_level state:" ,state)
        print("train_level goal:" ,goal)
        print("train_level subgoal_testing:" ,subgoal_testing)
    action_missed = False

    for step in range(H):
        done = False
        action = sample(policies[lvl], state, goal, subgoal_testing, lvl)

        if lvl >0 : # if not at the bottom layer
            
            # decide whether we subgoal_test at the following lower level
            subgoal_testing_next = subgoal_testing or np.random.binomial(1,lamb,1)[0]
            # subgoal_testing_next = True #### delete this !!!
            
            state_next, total_reward, done = train_level(lvl-1, state, action, subgoal_testing_next, total_reward)
            
            # action_missed = (state_next != action).any()
            action_missed = not np.allclose(state_next, action, atol=1.0)
            if action_missed:
                reward = -1
            else:
                reward = 0
                replay_buffer[lvl].append( (state, action, reward, state_next, goal, 0) ) ###
                return state_next, total_reward, done
                # done = True

#####            goals[lvl] = state
            
        else:
            # take a step in the real game
            # print('action is ', action)
            # print('about to take step. action is ', action)
            state_next, reward, done, _ = env.step(action)
            # print('...then')
            # print('state_next: ', state_next, ' reward: ', reward, ' done: ', done)
            total_reward += reward

        if lvl > 0:
            if action_missed: # action is the same as the g_i-1
                if subgoal_testing_next:
                    replay_buffer[lvl].append((state, action, -H, state_next, goal, 0))##########33
                    pass
                action = state_next #########3
        
        replay_buffer[lvl].append( (state, action, reward, state_next, goal, gamma if action_missed else 0) ) ###
        her_storage[lvl].append( (state, action, None, state_next, None, None ) )

#########
        # state_goal = np.concatenate((state, goal))
        # policy_input = torch.from_numpy(np.expand_dims(state_goal, axis=0)).float()

        # state_action_goal = np.concatenate((state, sample(policies[lvl], state_next, goal, False, lvl).reshape((-1,)), goal))
        # Q_input = torch.from_numpy(np.expand_dims(state_action_goal, axis=0)).float()

        # test_pol = policies[lvl].forward(policy_input).detach().numpy()[0]
        # test_Q = Q_networks[lvl](Q_input)

        # if episode%200 == 0 and lvl==0:
        #     print(' ')
        #     print('lvl: ', lvl, 'step: ', step)
        #     print('pol', test_pol)
        #     print('Q', test_Q)
        #     print('action taken', action)
        #     print('## state_next', state_next)
        #     print(' ')

#########

        state = state_next

        if len(replay_buffer[lvl]) > 4000:
            replay_buffer[lvl] = replay_buffer[lvl][-4000:]
        
        if done:
            break
            # return state_next, total_reward, done #########used to be break
    
    replay_buffer[lvl].extend(her_storage[lvl](all_goals=[]))
    return state_next, total_reward, done


def sample(policy, state, goal, subgoal_testing, lvl):
    state_goal = np.concatenate((state, goal))
    policy_input = torch.from_numpy(np.expand_dims(state_goal, axis=0)).float()

    if lvl==0:
        if subgoal_testing:
            action = np.argmax(policy.forward(policy_input).detach().numpy()[0])
        else:
            action = np.random.choice( 2, 1, p=F.softmax(policy.forward(policy_input)).detach().numpy()[0] )[0]
    else:
        if subgoal_testing:
            action = policy.forward( policy_input ).detach().numpy()[0]
        else:
            action = policy.forward(policy_input).detach().numpy()[0] + np.array([random.random() - 0.5 for i in range(4)])

        # action = top_goal
    return action


num_episodes=5000
num_levels = 3
top_goal = np.array([0,0,0,0])
H=30 #40
lamb = 0.1

learning_rate = 0.001

replay_buffer = [[], [], []]
her_storage = [[], [], []]
her_storage = [HER(), HER(), HER()]

gamma = 0.9

policies =[]
Q_networks = []
p_optimizers = []
q_optimizers = []

# initialising policy networks and Q networks
for i in range(num_levels):
    if i==0:
        pol = Policy(8, 2, 10)
        q_n = Policy(4+1+4, 1, 10) # we use the same network class but with different shape
    else:
        pol = Policy(8, 4, 10) # input dim is state_dim + goal_dim
        q_n = Policy(4+4+4, 1, 10) # input dim is state_dim + action_dim + goal_dim
    policies.append(pol)
    Q_networks.append(q_n)
    p_optimizers.append(optim.Adam(pol.parameters(), lr=learning_rate))
    q_optimizers.append(optim.Adam(q_n.parameters(), lr=learning_rate))



def update_parameters():
    tau = 0.01 # hyperparameter update rate for target network 0 <   1
    N = 8# hyperparameter number of instances we sample

    for lvl in range(num_levels): # iterate through all levels
        if len(replay_buffer[lvl]) <= N:
            break
        # sample N instances from the replay_buffer[i]
        # training_batch = replay_buffer[lvl].sample(num_instances = N)
        training_batch = random.sample(replay_buffer[lvl], N)
        y = []
        Qi_vec = []

        Q_lvl = Q_lvl_prime = Q_networks[lvl] # for now
        pi_lvl = pi_lvl_prime = policies[lvl] # for now
        Q_opt_lvl = q_optimizers[lvl]
        p_opt_lvl = p_optimizers[lvl]

        for state, action, reward, next_state, goal, gamma in training_batch:

            state_action_goal = np.concatenate((state, action.reshape((-1,)), goal))
            net_input = torch.from_numpy(np.expand_dims(state_action_goal, axis=0)).float()

            Qi_vec.append( Variable(Q_lvl(net_input), requires_grad=True) )

            state_action_goal = np.concatenate((state, sample(pi_lvl, next_state, goal, False, lvl).reshape((-1,)), goal))
            net_input = torch.from_numpy(np.expand_dims(state_action_goal, axis=0)).float()

            yn = Variable( reward + Q_lvl(net_input), requires_grad=True ) # turn this shit into a tensor

##
            # if lvl == 0:
            #     yn  = reward + Q_lvl(next_state, sample(pi_lvl_prime(next_state,goal)), goal)  # pi_i outputs the logits for a Bernoulli Distribution
            # else: # (lvl > 0)
            #     yn  = reward + Q_lvl(next_state, pi_lvl_prime(next_state,goal) + epsilon, goal),
            #     #where epsilon is sampled from the Gaussian Noise as consistent with the behavioural policy added noise
##
            y.append(yn)

        #Lcritic = 1/abs(B) SUM( FROM n=1 TO |B|,  (yn - Qlvl(sn , an, gn))^2 ) # |B| is the batch size

        Q_opt_lvl.zero_grad()
        critic_loss = nn.MSELoss()
        Lcritic = critic_loss( torch.tensor(y, requires_grad=True), torch.tensor(Qi_vec, requires_grad=True) )
        Lcritic.backward()
        Q_opt_lvl.step()


        Lactor = torch.tensor([0.0], requires_grad=True, dtype=torch.float)
        if lvl > 0: # not bottom level
            # Lactor = SUM( FROM n=1 TO |B|,  exp( -(pi_lvl(sn) -an )^2) *Qi(sn, an, gn) ) # Gaussian distribution of action (subgoal) vector
            i = 0
            for state, action, reward, next_state, goal, gamma in training_batch:
                diff = torch.tensor( sample(pi_lvl, state, goal, False, lvl) - action, requires_grad=True, dtype=torch.float)
                Lactor = Lactor - torch.exp(-diff.dot(diff)) * Qi_vec[i]
                i +=1
        else: # bottom level lvl = 0
            i = 0
            for state, action, reward, next_state, goal, gamma in training_batch:

                state_goal = np.concatenate((state, goal))
                policy_input = torch.from_numpy(np.expand_dims(state_goal, axis=0)).float()
                logits = pi_lvl(policy_input)
                probs = F.softmax(logits).squeeze()

                state_action_goal = np.concatenate((state, action.reshape((-1,)), goal))
                net_input = torch.from_numpy(np.expand_dims(state_action_goal, axis=0)).float()

###
                weights = []
                for act in reversed(range(2)): #all_actions: # how do we do this? ##########
                    ac = np.int64(act)
                    state_action_goal = np.concatenate((state, ac.reshape((-1,)), goal))
                    net_input = torch.from_numpy(np.expand_dims(state_action_goal, axis=0)).float()
                    Q_a = weights.append( Q_lvl(net_input) )

                Lactor = Lactor - torch.dot( probs, torch.tensor(weights, requires_grad=True) ) 
###

                # Lactor = Lactor - torch.sum(probs)*torch.tensor(Qi_vec[i], requires_grad=True, dtype=torch.float)
                # print(Q_lvl(net_input).shape)
                # Lactor = Lactor - probs.dot( Q_lvl(net_input) )
                i +=1

            # Lactor = - SUM( FROM n=1 TO |B|
            #                 SUM( FROM eta=0 TO |A|-1,
            #                     probs[eta] * Q_lvl(sn, eta, gn)
            #                 )
            #             )

        p_opt_lvl.zero_grad()
        Lactor.backward()
        p_opt_lvl.step()

        # trainable_parameters: List[tf.Variable] = self.model.trainable_variables()
        # optimizer.update_parameters(trainable_parameters, Lcritic)
        # optimizer.update_parameters(trainable_parameters, Lactor)
        # Updated the network for level i. Now it’s time to update the target network
        # Q_lvl_prime.trainableVariables  = (1-tau) * Q_lvl_prime.trainableVariables  + tau * Q_lvl.trainableVariables # ?
        # pi_lvl_prime.trainableVariables = (1-tau) * pi_lvl_prime.trainableVariables + tau * pi_lvl.trainableVariables # ?


rewards = []
# main loop
for episode in range(num_episodes):
    env = gym.make('CartPole-v0')
    state = env.reset()
    _, total_reward, _ = train_level(2, state, top_goal, False, 0)
    print('ep ',episode,', total_reward: ', total_reward)
    rewards.append(total_reward)

    update_parameters()

env.close()

smoothed_rewards = pd.Series.rolling(pd.Series(rewards), 10).mean()
smoothed_rewards = [elem for elem in smoothed_rewards]

plt.plot(rewards)
plt.plot(smoothed_rewards)
plt.plot()
plt.show()


