import numpy as np
from collections import deque
import random

class NeuralNetwork:

    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layer = {
                'weights': np.random.normal(0, 0.1, (sizes[i], sizes[i+1])),
                'biases': np.zeros((1, sizes[i+1]))
            }
            self.layers.append(layer)

        self.activation = activation

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return x

    def forward(self, x):
        current_input = x

        for i, layer in enumerate(self.layers):
            z = np.dot(current_input, layer['weights']) + layer['biases']
            if i == len(self.layers) - 1:
                current_input = z
            else:
                current_input = self._activate(z)

        return current_input

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer['weights'].flatten())
            params.extend(layer['biases'].flatten())
        return np.array(params)

    def set_params(self, params):
        idx = 0
        for layer in self.layers:
            w_size = layer['weights'].size
            b_size = layer['biases'].size

            layer['weights'] = params[idx:idx + w_size].reshape(layer['weights'].shape)
            idx += w_size

            layer['biases'] = params[idx:idx + b_size].reshape(layer['biases'].shape)
            idx += b_size

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:

    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.005, buffer_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.actor = NeuralNetwork(state_dim, [400, 300], action_dim, 'relu')
        self.actor_target = NeuralNetwork(state_dim, [400, 300], action_dim, 'relu')

        self.critic = NeuralNetwork(state_dim + action_dim, [400, 300], 1, 'relu')
        self.critic_target = NeuralNetwork(state_dim + action_dim, [400, 300], 1, 'relu')

        self.actor_target.set_params(self.actor.get_params())
        self.critic_target.set_params(self.critic.get_params())

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.noise_std = 0.2

    def select_action(self, state, add_noise=True):
        state = state.reshape(1, -1)
        action = self.actor.forward(state).flatten()

        if add_noise:
            noise = np.random.normal(0, self.noise_std, self.action_dim)
            action = action + noise

        action = np.clip(action, -1, 1)
        return action

    def update_target_networks(self):
        actor_params = self.actor.get_params()
        actor_target_params = self.actor_target.get_params()
        new_actor_target = self.tau * actor_params + (1 - self.tau) * actor_target_params
        self.actor_target.set_params(new_actor_target)

        critic_params = self.critic.get_params()
        critic_target_params = self.critic_target.get_params()
        new_critic_target = self.tau * critic_params + (1 - self.tau) * critic_target_params
        self.critic_target.set_params(new_critic_target)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        next_actions = self.actor_target.forward(next_states)
        next_q_inputs = np.concatenate([next_states, next_actions], axis=1)
        next_q_values = self.critic_target.forward(next_q_inputs).flatten()
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q_inputs = np.concatenate([states, actions], axis=1)
        current_q = self.critic.forward(current_q_inputs).flatten()

        critic_loss = np.mean((current_q - target_q) ** 2)

        pred_actions = self.actor.forward(states)
        actor_q_inputs = np.concatenate([states, pred_actions], axis=1)
        actor_loss = -np.mean(self.critic.forward(actor_q_inputs))

        self._update_critic(states, actions, target_q)
        self._update_actor(states)

        self.update_target_networks()

        return critic_loss, actor_loss

    def _update_critic(self, states, actions, targets):
        inputs = np.concatenate([states, actions], axis=1)
        predictions = self.critic.forward(inputs).flatten()

        error = np.mean((predictions - targets) ** 2)
        grad = error * self.lr_critic

        current_params = self.critic.get_params()
        self.critic.set_params(current_params - grad * np.sign(current_params) * 0.001)

    def _update_actor(self, states):
        current_params = self.actor.get_params()

        grad = np.random.normal(0, self.lr_actor, current_params.shape)
        self.actor.set_params(current_params + grad * 0.001)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)