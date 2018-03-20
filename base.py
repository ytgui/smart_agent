import time
import numpy as np


class AgentBase:
    def __init__(self):
        # sub module
        self._env = None
        self._replay = None
        # epsilon greedy
        self._epsilon = 0.0
        self._epsilon_min = 0.0
        self._epsilon_max = 0.8
        # global_step
        self._global_episode = 0
        self._global_step = 0

    def is_not_used(self):
        pass

    def _epsilon_greedy(self):
        # (0, 0) (10, 0.2) (30, 0.55) (50, 0.75) (60, 0.8)
        self._epsilon = np.tanh(0.02 * self._global_episode)
        self._epsilon = np.maximum(self._epsilon, self._epsilon_min)
        self._epsilon = np.minimum(self._epsilon, self._epsilon_max)
        return self._epsilon

    def _train_impl(self, max_step):
        # prepare for epsilon greedy
        self._global_episode += 1
        # train
        state = self._env.reset()
        for step in range(max_step):
            self._global_step += 1
            # 1. predict
            action = self._explore(state)
            # 2. action
            state_, reward, done, info = self._env.step(action)
            reward = -1.0 if done else reward
            # 3. perceive
            self._perceive(state, action, state_, reward, done)
            # 4. update
            state = state_
            if done:
                break

    def _test_impl(self, max_step, delay, gui):
        # test
        total_reward = 0
        state = self._env.reset()
        for step in range(max_step):
            if gui:
                self._env.render()
            time.sleep(delay)
            # 1. predict
            action = self._optimal_action(state)
            # 2. action
            state_, reward, done, info = self._env.step(action)
            # 3. perceive
            total_reward += reward
            # 4. update
            state = state_
            if done:
                break
        return total_reward

    def _explore(self, state):
        epsilon = self._epsilon_greedy()
        if np.random.uniform() > epsilon:
            action = self._random_action()
        else:
            action = self._optimal_action(state)
        return action

    def _random_action(self):
        self.is_not_used()
        raise NotImplementedError('function must be overridden.')

    def _optimal_action(self, state):
        self.is_not_used()
        raise NotImplementedError('function must be overridden.')

    def _perceive(self, state, action, state_, reward, done):
        self.is_not_used()
        raise NotImplementedError('function must be overridden.')
