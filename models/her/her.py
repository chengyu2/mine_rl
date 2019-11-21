import random
from typing import Tuple


class HER:

    def __init__(self, epsilon: int=0, strategy='future'):
        """

        :param epsilon:
        :param strategy: String, expected one of ['future', 'standard']
        """
        self._epsilon = epsilon
        self._replay_buffer = [] # TODO: improve to queue
        self.strategy = strategy
        allowed_strategies = {'future', 'standard'}
        assert strategy in allowed_strategies, f"Expected one of {allowed_strategies}"

    def clear(self):
        self._replay_buffer = []

    def append(self, transition: Tuple):
        self._replay_buffer.append(transition)

    def compare_state(self, state_a, state_b):
        # TODO: add a function here based on state transitions
        return False

    def is_in_states(self, state_a, states):
        for state_b in states:
            if self.compare_state(state_a, state_b):
                return True
        return False

    def get_reward(self, s, g):
        return 1

    def get_hindsight_goal(self, i):
        if self.strategy == 'future':
            _, _, _, g, _, _ = self._replay_buffer[random.randint(i, len(self._replay_buffer))]
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




