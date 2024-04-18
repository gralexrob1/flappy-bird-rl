import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm


class BaseAgent:

    def __init__(self, env, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.policy = None

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state])
        return action

    def epsilon_greedy_action_from_Q(self, state):
        if tuple(state) in self.Q:
            action_probs = np.ones(self.env.action_space.n) * self.epsilon
            action_probs /= self.env.action_space.n
            best_action = self.argmax(self.Q[state])
            action_probs[best_action] = action_probs[best_action] + 1 - self.epsilon
            action = np.random.choice(self.env.action_space.n, p=action_probs)
        else:
            action = self.env.action_space.sample()
        return action

    def argmax(self, Q_values):
        """Argmax with random tie-breaking"""
        top = float("-inf")
        ties = []

        for i in range(len(Q_values)):
            if Q_values[i] > top:
                top = Q_values[i]
                ties = []

            if Q_values[i] == top:
                ties.append(i)

        return random.choice(ties)

    def train(self):
        raise NotImplementedError("Train needs to be implemented in child class")

    def test(self, num_episode, max_reward=10_000):
        if self.policy is None:
            print("### Random Agent ###")

        rewards = []
        for _ in tqdm(range(num_episode)):
            state, info = self.env.reset()
            total_reward = 0
            while True:
                if self.policy is None:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy[state] if state in self.policy else 0
                next_state, reward, done, _, info = self.env.step(action)
                total_reward += reward
                state = next_state
                if total_reward == max_reward or done:
                    break
            rewards.append(total_reward)
        return np.array(rewards)


class MonteCarloAgent(BaseAgent):

    def __init__(self, env, alpha, gamma, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        super().__init__(env, gamma, epsilon, epsilon_decay, epsilon_min)
        self.alpha = alpha
        self.N = defaultdict(int)
        self.S = defaultdict(float)
        self.V = defaultdict(float)

    def generate_episode_from_Q(self, max_reward=10_000):
        episode = []
        state, info = self.env.reset()
        total_reward = 0
        while True:
            action = self.epsilon_greedy_action_from_Q(state)
            next_state, reward, done, _, info = self.env.step(action)
            total_reward += reward
            episode.append((state, action, reward))
            state = next_state
            if total_reward == max_reward or done:
                break
        return episode

    def update_Q(self, episode):
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.gamma**i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            old_Q = self.Q[state][actions[i]]
            G_i = sum(rewards[i:] * discounts[: -i - 1])
            self.Q[state][actions[i]] = old_Q + self.alpha * (G_i - old_Q)

    def train(self, num_episode, max_reward=10_000):
        epsilons = []
        episode_rewards = []
        episode_lengths = []
        for _ in tqdm(range(num_episode)):
            epsilons.append(self.epsilon)
            episode = self.generate_episode_from_Q(max_reward)
            self.update_Q(episode)
            self.decay_epsilon()
            episode_rewards.append(sum(r for (_, _, r) in episode))
            episode_lengths.append(len(episode))
        self.policy = dict((k, np.argmax(v)) for k, v in self.Q.items())
        return epsilons, episode_rewards, episode_lengths


class SarsaAgent(BaseAgent):

    def __init__(self, env, alpha, gamma, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        super().__init__(env, gamma, epsilon, epsilon_decay, epsilon_min)
        self.alpha = alpha

    def update_Q(self, state, action, reward, next_state, next_action):
        Q = self.Q[state][action]
        next_Q = self.Q[next_state][next_action]
        target = reward + self.gamma * next_Q
        self.Q[state][action] = Q + self.alpha * (target - Q)

    def train(self, num_episode, max_reward=10_000):
        epsilons = []
        rewards = []
        for _ in tqdm(range(num_episode)):
            epsilons.append(self.epsilon)
            state, info = self.env.reset()
            action = self.epsilon_greedy_action(state)
            episode_reward = 0
            while True:
                next_state, reward, done, _, info = self.env.step(action)
                next_action = self.epsilon_greedy_action(next_state)
                self.update_Q(state, action, reward, next_state, next_action)
                episode_reward += reward
                state = next_state
                action = next_action
                if episode_reward == max_reward or done:
                    break
            self.decay_epsilon()
            rewards.append(episode_reward)
        self.policy = dict((k, np.argmax(v)) for k, v in self.Q.items())
        return np.array(epsilons), np.array(rewards)


class SarsaNstepAgent(BaseAgent):
    def __init__(self, env, alpha, gamma, n, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        super().__init__(env, gamma, epsilon, epsilon_decay, epsilon_min)
        self.alpha = alpha
        self.n = n

    def update_Q(self, tau, T, states_buffer, actions_buffer, rewards_buffer):
        G = np.sum(
            [
                self.gamma ** (i - tau - 1) * rewards_buffer[i]
                for i in range(tau + 1, min(tau + self.n, T))
            ]
        )
        if tau + self.n < T:
            G += (
                self.gamma**self.n
                * self.Q[states_buffer[tau + self.n]][actions_buffer[tau + self.n]]
            )
        Q = self.Q[states_buffer[tau]][actions_buffer[tau]]
        target = G
        self.Q[states_buffer[tau]][actions_buffer[tau]] = Q + self.alpha * (target - Q)

    def train(self, num_episode, max_reward=5_000):
        epsilons = []
        episode_rewards = []
        episode_lengths = []

        for episode in tqdm(range(num_episode)):
            epsilons.append(self.epsilon)
            total_reward = 0

            rewards_buffer = []
            states_buffer = []
            actions_buffer = []

            state, info = self.env.reset()
            states_buffer.append(state)

            action = self.epsilon_greedy_action(state)
            actions_buffer.append(action)

            T = np.inf
            t = 0
            tau = 0

            assert len(states_buffer) == len(actions_buffer)

            while tau < (T - 1):

                if t < T:
                    next_state, reward, done, _, info = self.env.step(action)
                    total_reward += reward
                    rewards_buffer.append(reward)
                    states_buffer.append(next_state)

                    if total_reward == max_reward or done:
                        T = t + 1
                    else:
                        next_action = self.epsilon_greedy_action(next_state)
                        actions_buffer.append(next_action)

                tau = t - self.n + 1

                if tau >= 0:
                    self.update_Q(tau, T, states_buffer, actions_buffer, rewards_buffer)

                t += 1
                state = next_state
                action = next_action

            self.decay_epsilon()
            episode_rewards.append(np.sum(rewards_buffer))
            episode_lengths.append(len(rewards_buffer))

        self.policy = dict((k, np.argmax(v)) for k, v in self.Q.items())

        return np.array(epsilons), np.array(episode_rewards), np.array(episode_lengths)


class SarsaLambdaAgent(BaseAgent):
    def __init__(self, env, alpha, gamma, lambd, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        super().__init__(env, gamma, epsilon, epsilon_decay, epsilon_min)
        self.alpha = alpha
        self.lambd = lambd

    def update_Q(self, state, action, reward, next_state, next_action, z):
        delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        z[state][action] += 1

        for s in self.Q.keys():
            for a in range(self.env.action_space.n):
                self.Q[s][a] += self.alpha * delta * z[s][a]
                z[s][a] *= self.gamma * self.lambd

    def train(self, num_episode, max_reward=10_000):
        epsilons = []
        episode_rewards = []
        episode_lengths = []

        for episode in tqdm(range(num_episode)):
            epsilons.append(self.epsilon)
            total_reward = 0

            rewards_buffer = []
            states_buffer = []
            actions_buffer = []

            state, info = self.env.reset()
            states_buffer.append(state)

            action = self.epsilon_greedy_action(state)
            actions_buffer.append(action)

            z = defaultdict(lambda: np.zeros(self.env.action_space.n))

            while True:
                next_state, reward, done, _, info = self.env.step(action)
                total_reward += reward
                rewards_buffer.append(reward)
                states_buffer.append(next_state)

                next_action = self.epsilon_greedy_action(next_state)
                actions_buffer.append(next_action)

                self.update_Q(state, action, reward, next_state, next_action, z)

                state = next_state
                action = next_action

                if total_reward == max_reward or done:
                    break

            self.decay_epsilon()
            episode_rewards.append(np.sum(rewards_buffer))
            episode_lengths.append(len(rewards_buffer))

        self.policy = dict((k, np.argmax(v)) for k, v in self.Q.items())

        return np.array(epsilons), np.array(episode_rewards), np.array(episode_lengths)
