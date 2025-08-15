import numpy as np

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

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lr = lr

        self.policy_net = NeuralNetwork(state_dim, [256, 256], action_dim, 'tanh')

        self.value_net = NeuralNetwork(state_dim, [256, 256], 1, 'relu')

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state):
        state = state.reshape(1, -1)

        action_mean = self.policy_net.forward(state).flatten()

        action_std = 0.5 
        action = action_mean + np.random.normal(0, action_std, self.action_dim)
        action = np.clip(action, -1, 1)

        log_prob = -0.5 * np.sum((action - action_mean) ** 2) / (action_std ** 2)

        value = self.value_net.forward(state).flatten()[0]

        return action, log_prob, value

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self, next_value=0):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        advantage = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * 0.95 * (1 - dones[t]) * advantage  # λ = 0.95
            advantages[t] = advantage

        returns = advantages + values[:-1]

        return advantages, returns

    def update(self, next_value=0, epochs=10):
        if len(self.states) == 0:
            return

        advantages, returns = self.compute_advantages(next_value)

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        old_values = np.array(self.values)

        total_policy_loss = 0
        total_value_loss = 0

        for epoch in range(epochs):
            action_means = self.policy_net.forward(states)

            action_std = 0.5
            new_log_probs = np.array([
                -0.5 * np.sum((actions[i] - action_means[i]) ** 2) / (action_std ** 2)
                for i in range(len(actions))
            ])

            ratios = np.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = np.clip(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -np.mean(np.minimum(surr1, surr2))

            new_values = self.value_net.forward(states).flatten()
            value_loss = np.mean((new_values - returns) ** 2)

            self._update_policy(states, actions, advantages, ratios)
            self._update_value(states, returns)

            total_policy_loss += policy_loss
            total_value_loss += value_loss

        self.clear_trajectory()

        return total_policy_loss / epochs, total_value_loss / epochs

    def _update_policy(self, states, actions, advantages, ratios):
        clipped_ratios = np.clip(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        current_params = self.policy_net.get_params()

        avg_advantage = np.mean(advantages)
        grad_direction = np.sign(avg_advantage) * self.lr

        noise = np.random.normal(0, 0.001, current_params.shape)
        self.policy_net.set_params(current_params + grad_direction * noise)

    def _update_value(self, states, returns):
        predictions = self.value_net.forward(states).flatten()
        error = np.mean((predictions - returns) ** 2)

        current_params = self.value_net.get_params()
        grad = error * self.lr

        self.value_net.set_params(current_params - grad * np.sign(current_params) * 0.001)

    def clear_trajectory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
