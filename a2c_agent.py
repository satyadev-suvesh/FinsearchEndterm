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
            if i == len(self.layers) - 1:  # Output layer
                current_input = z  # Linear activation for output
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

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.entropy_coef = entropy_coef

        self.actor = NeuralNetwork(state_dim, [256, 256], action_dim, 'tanh')

        self.critic = NeuralNetwork(state_dim, [256, 256], 1, 'relu')

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def select_action(self, state):
        state = state.reshape(1, -1)

        action_mean = self.actor.forward(state).flatten()

        action_std = 0.5
        action = action_mean + np.random.normal(0, action_std, self.action_dim)
        action = np.clip(action, -1, 1)

        log_prob = -0.5 * np.sum((action - action_mean) ** 2) / (action_std ** 2)

        value = self.critic.forward(state).flatten()[0]

        return action, log_prob, value

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_advantages(self):

        advantages = []
        returns = []

        for i in range(len(self.rewards)):
            if self.dones[i]:
                next_value = 0
            else:
                next_value = self.critic.forward(self.next_states[i].reshape(1, -1)).flatten()[0]

            advantage = self.rewards[i] + self.gamma * next_value - self.values[i]
            advantages.append(advantage)

            returns.append(self.rewards[i] + self.gamma * next_value)

        return np.array(advantages), np.array(returns)

    def update(self):

        if len(self.states) == 0:
            return

        advantages, returns = self.compute_advantages()

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)
        values = np.array(self.values)

        actor_loss = -np.mean(log_probs * advantages)

        predicted_values = self.critic.forward(states).flatten()
        critic_loss = np.mean((predicted_values - returns) ** 2)

        self._update_actor(states, actions, advantages)

        self._update_critic(states, returns)

        self.clear_buffers()

        return actor_loss, critic_loss

    def _update_actor(self, states, actions, advantages):

        action_means = self.actor.forward(states)

        current_params = self.actor.get_params()

        avg_advantage = np.mean(advantages)

        gradient = avg_advantage * self.lr_actor
        noise = np.random.normal(0, 0.001, current_params.shape)

        self.actor.set_params(current_params + gradient * noise)

    def _update_critic(self, states, returns):
        current_values = self.critic.forward(states).flatten()

        value_error = np.mean((current_values - returns) ** 2)

        current_params = self.critic.get_params()
        gradient = value_error * self.lr_critic

        self.critic.set_params(current_params - gradient * np.sign(current_params) * 0.001)

    def clear_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []
