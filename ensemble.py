import numpy as np
import pandas as pd

from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
from a2c_agent import A2CAgent


def calculate_technical_indicators(df):
    close_col = [col for col in df.columns if 'close' in col.lower().strip()][0]
    high_col = [col for col in df.columns if 'high' in col.lower().strip()][0]
    low_col = [col for col in df.columns if 'low' in col.lower().strip()][0]
    
    close_prices = df[close_col]
    high_prices = df[high_col]
    low_prices = df[low_col]

    ema12 = close_prices.ewm(span=12).mean()
    ema26 = close_prices.ewm(span=26).mean()
    macd = ema12 - ema26
    
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    tp = (high_prices + low_prices + close_prices) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma_tp) / (0.015 * (mad + 1e-8))
    
    high_diff = high_prices.diff()
    adx = high_diff.rolling(window=14).mean().abs()  
    
    macd = macd.fillna(0)
    rsi = rsi.fillna(50)
    cci = cci.fillna(0)
    adx = adx.fillna(25)
    
    return macd.values, rsi.values, cci.values, adx.values


class StockTradingEnv:
    def __init__(self, price_data, indicator_data, initial_balance=1e6, transaction_cost=0.001):
        self.price_data = price_data
        self.indicator_data = indicator_data
        self.num_stocks = price_data.shape[1]
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares = 100

        self.state_dim = 1 + self.num_stocks + self.num_stocks + 4 * self.num_stocks
        self.action_dim = self.num_stocks

        self.current_step = 0
        self.balance = None
        self.holdings = None
        self.prices = None
        self.macd = None
        self.rsi = None
        self.cci = None
        self.adx = None

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_stocks)

        self.prices = self.price_data[self.current_step]
        self.macd = self.indicator_data['macd'][self.current_step]
        self.rsi = self.indicator_data['rsi'][self.current_step]
        self.cci = self.indicator_data['cci'][self.current_step]
        self.adx = self.indicator_data['adx'][self.current_step]

        return self._get_state()

    def _get_state(self):
        state = np.concatenate([
            [self.balance / self.initial_balance],
            self.prices / 30000.0,
            self.holdings / 100.0,
            self.macd / 100.0,
            self.rsi / 100.0,
            self.cci / 200.0,
            self.adx / 100.0
        ])
        return state.astype(np.float32)

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        shares_to_trade = (actions * self.max_shares).astype(int)

        portfolio_value_before = self.balance + np.sum(self.holdings * self.prices)
        total_transaction_cost = 0.0

        for i, shares in enumerate(shares_to_trade):
            if shares > 0:
                cost = shares * self.prices[i] * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.holdings[i] += shares
                    total_transaction_cost += cost * self.transaction_cost
            elif shares < 0:
                sell_shares = min(-shares, self.holdings[i])
                if sell_shares > 0:
                    revenue = sell_shares * self.prices[i] * (1 - self.transaction_cost)
                    self.balance += revenue
                    self.holdings[i] -= sell_shares
                    total_transaction_cost += sell_shares * self.prices[i] * self.transaction_cost

        self.current_step += 1
        done = False
        if self.current_step >= len(self.price_data):
            done = True
            self.current_step = len(self.price_data) - 1

        self.prices = self.price_data[self.current_step]
        self.macd = self.indicator_data['macd'][self.current_step]
        self.rsi = self.indicator_data['rsi'][self.current_step]
        self.cci = self.indicator_data['cci'][self.current_step]
        self.adx = self.indicator_data['adx'][self.current_step]

        portfolio_value_after = self.balance + np.sum(self.holdings * self.prices)
        reward = (portfolio_value_after - portfolio_value_before) - total_transaction_cost
        reward /= self.initial_balance

        if portfolio_value_after <= 0.1 * self.initial_balance:
            done = True

        next_state = self._get_state()
        info = {"portfolio_value": portfolio_value_after}
        return next_state, reward, done, info


class EnsembleStrategy:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ddpg_agent = DDPGAgent(state_dim, action_dim)
        self.ppo_agent = PPOAgent(state_dim, action_dim)
        self.a2c_agent = A2CAgent(state_dim, action_dim)

        self.agent_performances = {'DDPG': [], 'PPO': [], 'A2C': []}

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        if len(returns) == 0:
            return float('-inf')
        daily_rf = risk_free_rate / 252
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8
        sharpe = (mean_ret - daily_rf) / std_ret
        return sharpe

    def select_best_agent(self, validation_returns):
        sharpe_scores = {}
        for agent_name, rets in validation_returns.items():
            sharpe_scores[agent_name] = self.calculate_sharpe_ratio(rets)
        best_agent = max(sharpe_scores, key=sharpe_scores.get)
        print(f"Validation Sharpe Ratios: {sharpe_scores}")
        print(f"Selected Best Agent: {best_agent}")
        return best_agent, sharpe_scores

    def get_agent(self, name):
        agents = {'DDPG': self.ddpg_agent, 'PPO': self.ppo_agent, 'A2C': self.a2c_agent}
        return agents[name]

    def train_all_agents(self, env, episodes):
        print(f"Training all agents for {episodes} episodes...")
        all_returns = {'DDPG': [], 'PPO': [], 'A2C': []}

        for ep in range(episodes):
            if ep % max(1, episodes // 5) == 0:
                print(f"Episode {ep+1}/{episodes}")
            
            for name in all_returns.keys():
                env_copy = StockTradingEnv(env.price_data, env.indicator_data)
                agent = self.get_agent(name)

                state = env_copy.reset()
                total_reward = 0.0
                steps = 0
                max_steps = min(50, len(env.price_data) - 1)

                while steps < max_steps:
                    if name == 'DDPG':
                        action = agent.select_action(state)
                        next_state, reward, done, _ = env_copy.step(action)
                        agent.add_experience(state, action, reward, next_state, done)
                        if len(agent.replay_buffer) > 16:
                            agent.train()
                    elif name == 'PPO':
                        action, log_prob, val = agent.select_action(state)
                        next_state, reward, done, _ = env_copy.step(action)
                        agent.store_transition(state, action, reward, log_prob, val, done)
                        if len(agent.states) >= 8:
                            agent.update()
                    elif name == 'A2C':
                        action, log_prob, val = agent.select_action(state)
                        next_state, reward, done, _ = env_copy.step(action)
                        agent.store_transition(state, action, reward, next_state, done, log_prob, val)
                        if len(agent.states) >= 4:
                            agent.update()

                    state = next_state
                    total_reward += reward
                    steps += 1
                    if done:
                        break
                
                all_returns[name].append(total_reward)
        return all_returns

    def quarterly_rebalance(self, env, training_episodes=10, validation_episodes=3):
        print("\nStarting quarterly rebalance...")
        training_returns = self.train_all_agents(env, training_episodes)
        validation_returns = {k: training_returns[k][-validation_episodes:] for k in training_returns}
        best_agent, sharpe_scores = self.select_best_agent(validation_returns)
        return best_agent, sharpe_scores, training_returns


def load_data(path):
    print(f"Loading data from {path}...")
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: File '{path}' not found!")
        return None, None
    
    df.columns = df.columns.str.strip()
    
    print(f"Available columns: {df.columns.tolist()}")
    
    close_col = None
    for col in df.columns:
        if 'close' in col.lower():
            close_col = col
            break
    
    if close_col is None:
        print("Error: No 'Close' column found in CSV!")
        return None, None
    
    prices = df[close_col].values.reshape(-1, 1)
    
    try:
        macd, rsi, cci, adx = calculate_technical_indicators(df)
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return None, None
    
    indicators = {
        'macd': macd.reshape(-1, 1),
        'rsi': rsi.reshape(-1, 1),
        'cci': cci.reshape(-1, 1),
        'adx': adx.reshape(-1, 1)
    }
    
    print(f"Loaded {len(prices)} timesteps, 1 stock")
    print(f"Price range: ₹{prices.min():.2f} - ₹{prices.max():.2f}")
    
    return prices, indicators


def example_usage():
    global np
    print("Starting Ensemble Trading Strategy with real OHLC data...")

    train_prices, train_indicators = load_data('3Years.csv')
    if train_prices is None:
        return None, None
        
    test_prices, test_indicators = load_data('test.csv')
    if test_prices is None:
        return None, None

    train_env = StockTradingEnv(train_prices, train_indicators)
    test_env = StockTradingEnv(test_prices, test_indicators)

    print(f"Training environment: {train_env.num_stocks} stock(s), {len(train_prices)} timesteps")
    print(f"Test environment: {test_env.num_stocks} stock(s), {len(test_prices)} timesteps")

    ensemble = EnsembleStrategy(train_env.state_dim, train_env.action_dim)
    best_agent, sharpe_scores, _ = ensemble.quarterly_rebalance(
        train_env, training_episodes=50, validation_episodes=10
    )

    print(f"\n=== Testing Best Agent: {best_agent} ===")
    agent = ensemble.get_agent(best_agent)

    state = test_env.reset()
    initial_value = test_env.balance + np.sum(test_env.holdings * test_env.prices)
    portfolio_values = [initial_value]

    print(f"Initial portfolio: ₹{initial_value:,.2f}")
    print("Trading progress:")

    for step in range(min(15, len(test_prices)-1)):
        if best_agent == 'DDPG':
            action = agent.select_action(state, add_noise=False)
        else:
            action, _, _ = agent.select_action(state)
        
        next_state, reward, done, info = test_env.step(action)
        portfolio_values.append(info['portfolio_value'])
        
        if step % 3 == 0:
            print(f"  Step {step+1}: Portfolio = ₹{info['portfolio_value']:,.2f}, "
                  f"Action = {action[0]:.3f}, Holdings = {test_env.holdings[0]}")
        
        state = next_state
        if done:
            print(f"  Episode ended at step {step+1}")
            break

    final_value = portfolio_values[-1]
    total_return = 100.0 * (final_value - initial_value) / initial_value

    print(f"\n=== FINAL RESULTS ===")
    print(f"Initial portfolio value: ₹{initial_value:,.2f}")
    print(f"Final portfolio value:   ₹{final_value:,.2f}")
    print(f"Total return:            {total_return:.2f}%")
    print(f"Best agent:              {best_agent}")
    print(f"Number of test steps:    {len(portfolio_values)-1}")

    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    cum_return = (final_value / initial_value) - 1
    n_years = len(portfolio_values) / 252  
    ann_return = (final_value / initial_value) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
    sharpe = (np.mean(returns) * 252 - 0.02) / (np.std(returns) * np.sqrt(252) + 1e-8) if len(returns) > 1 else 0
    stats = {'cum': cum_return, 'ann': ann_return, 'vol': vol, 'sharpe': sharpe}
    print(f"Strategy stats: {stats}")

    return ensemble, portfolio_values


if __name__ == '__main__':
    ensemble, results = example_usage()
