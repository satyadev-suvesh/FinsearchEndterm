import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent
from ppo_agent import PPOAgent
from a2c_agent import A2CAgent
import ta


class ImprovedNiftyTradingEnv:

    def __init__(self, price_data, technical_indicators, initial_balance=1000000, transaction_cost=0.0005):
        self.price_data = price_data
        self.technical_indicators = technical_indicators
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = 2000

        self.returns = np.diff(price_data) / price_data[:-1]
        self.returns = np.concatenate([[0], self.returns])

        self.sma_short = self._calculate_sma(price_data, 5)
        self.sma_long = self._calculate_sma(price_data, 20)

        self.state_dim = 10
        self.action_dim = 1

        self.reset()

    def _calculate_sma(self, prices, window):
        sma = np.zeros(len(prices))
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            sma[i] = np.mean(prices[start_idx:i+1])
        return sma

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        self.prev_portfolio_value = self.initial_balance

        return self._get_state()

    def _get_state(self):
        current_price = self.price_data[self.current_step]

        momentum_window = 3
        start_idx = max(0, self.current_step - momentum_window + 1)
        recent_returns = self.returns[start_idx:self.current_step + 1]
        price_momentum = np.mean(recent_returns) if len(recent_returns) > 0 else 0

        trend_signal = (self.sma_short[self.current_step] - self.sma_long[self.current_step]) / self.sma_long[self.current_step]

        vol_window = 10
        start_idx = max(0, self.current_step - vol_window + 1)
        recent_returns_vol = self.returns[start_idx:self.current_step + 1]
        volatility = np.std(recent_returns_vol) if len(recent_returns_vol) > 1 else 0

        state = np.array([
            (self.balance / self.initial_balance - 0.5) * 2,
            (current_price / np.mean(self.price_data) - 1),
            self.position / self.max_position,
            np.clip(price_momentum * 100, -1, 1),
            np.clip(trend_signal, -0.5, 0.5) * 2,
            np.clip(self.technical_indicators['MACD'][self.current_step] / 50.0, -1, 1),
            (self.technical_indicators['RSI'][self.current_step] - 50) / 50,
            np.clip(self.technical_indicators['CCI'][self.current_step] / 200.0, -1, 1),
            (self.technical_indicators['ADX'][self.current_step] - 25) / 50,
            np.clip(volatility * 100, 0, 1)
        ])

        return state.astype(np.float32)

    def step(self, action):
        action_val = np.clip(action, -1, 1)[0] if isinstance(action, np.ndarray) else np.clip(action, -1, 1)

        current_price = self.price_data[self.current_step]
        portfolio_value_before = self.balance + self.position * current_price

        desired_position_change = int(action_val * self.max_position * 0.5)

        total_cost = 0
        if desired_position_change != 0:
            trade_value = abs(desired_position_change) * current_price
            cost = trade_value * self.transaction_cost

            if desired_position_change > 0:
                max_affordable = int(self.balance / (current_price * (1 + self.transaction_cost)))
                actual_trade = min(desired_position_change, max_affordable)
                if actual_trade > 0:
                    self.balance -= actual_trade * current_price * (1 + self.transaction_cost)
                    self.position += actual_trade
                    total_cost = actual_trade * current_price * self.transaction_cost
            else: 
                actual_trade = max(desired_position_change, -self.position)
                if actual_trade < 0:
                    self.balance += abs(actual_trade) * current_price * (1 - self.transaction_cost)
                    self.position += actual_trade
                    total_cost = abs(actual_trade) * current_price * self.transaction_cost

        self.current_step += 1

        if self.current_step >= len(self.price_data) - 1:
            self.done = True
            next_price = current_price
        else:
            next_price = self.price_data[self.current_step]

        portfolio_value_after = self.balance + self.position * next_price

        basic_pnl = (portfolio_value_after - portfolio_value_before - total_cost) / self.initial_balance
        market_return = (next_price - current_price) / current_price if not self.done else 0
        position_reward = 0
        if self.position > 0 and market_return > 0: position_reward = 0.001
        elif self.position < 0 and market_return < 0: position_reward = 0.001
        elif abs(self.position) > 0 and market_return * np.sign(self.position) < 0: position_reward = -0.0005
        trend_reward = 0
        if not self.done:
            short_ma = self.sma_short[self.current_step]
            long_ma = self.sma_long[self.current_step]
            if short_ma > long_ma and self.position > 0: trend_reward = 0.0005
            elif short_ma < long_ma and self.position < 0: trend_reward = 0.0005
        trading_penalty = -abs(desired_position_change) / self.max_position * 0.0001
        reward = basic_pnl + position_reward + trend_reward + trading_penalty
        if portfolio_value_after > 0.5 * self.initial_balance: reward += 0.0001
        if portfolio_value_after <= 0.1 * self.initial_balance:
            self.done = True
            reward = -0.1

        self.prev_portfolio_value = portfolio_value_after

        return self._get_state(), reward, self.done, {'portfolio_value': portfolio_value_after}

class ImprovedEnsembleStrategy:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ddpg_agent = DDPGAgent(state_dim, action_dim)
        self.ppo_agent = PPOAgent(state_dim, action_dim)
        self.a2c_agent = A2CAgent(state_dim, action_dim)

        self.agent_performances = {'DDPG': [], 'PPO': [], 'A2C': []}
        self.current_best_agent = None

    def calculate_performance_metric(self, returns, benchmark_returns):
        if len(returns) == 0: return -np.inf
        returns_arr = np.array(returns)
        daily_rf = 0.02 / 252
        excess_returns = returns_arr - daily_rf
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        if len(benchmark_returns) == len(returns):
            benchmark_arr = np.array(benchmark_returns)
            alpha = np.mean(returns_arr - benchmark_arr)
        else:
            alpha = 0
        return sharpe + alpha * 10

    def get_agent(self, agent_name):
        agent_map = {'DDPG': self.ddpg_agent, 'PPO': self.ppo_agent, 'A2C': self.a2c_agent}
        if agent_name in agent_map:
            return agent_map[agent_name]
        raise ValueError(f"Unknown agent: {agent_name}")

    def train_all_agents(self, env, episodes=30):
        print(f"\nTraining all agents for {episodes} episodes")
        agent_returns = {'DDPG': [], 'PPO': [], 'A2C': []}
        max_steps = len(env.price_data) - 1

        for episode in range(episodes):
            if episode % max(1, episodes // 10) == 0:
                print(f"Episode {episode}/{episodes}")

            for agent_name in ['DDPG', 'PPO', 'A2C']:
                state = env.reset()
                episode_return = 0
                agent = self.get_agent(agent_name)

                for step in range(max_steps):
                    if agent_name == 'DDPG':
                        action = agent.select_action(state, add_noise=True)
                        next_state, reward, done, info = env.step(action)
                        agent.add_experience(state, action, reward, next_state, done)
                        if len(agent.replay_buffer) > 64: agent.train(64)
                    elif agent_name == 'PPO':
                        action, log_prob, value = agent.select_action(state)
                        next_state, reward, done, info = env.step(action)
                        agent.store_transition(state, action, reward, log_prob, value, done)
                        if len(agent.states) >= 32: agent.update()
                    elif agent_name == 'A2C':
                        action, log_prob, value = agent.select_action(state)
                        next_state, reward, done, info = env.step(action)
                        agent.store_transition(state, action, reward, next_state, done, log_prob, value)
                        if len(agent.states) >= 16: agent.update()
                    
                    state = next_state
                    episode_return += reward
                    if done: break
                
                agent_returns[agent_name].append(episode_return)
        return agent_returns

    def select_best_agent(self, validation_returns, benchmark_returns=None):
        performance_scores = {}
        for agent_name, returns in validation_returns.items():
            if benchmark_returns:
                score = self.calculate_performance_metric(returns, benchmark_returns)
            else:
                if len(returns) == 0: score = -np.inf
                else:
                    returns_arr = np.array(returns)
                    score = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8)
            performance_scores[agent_name] = score
        
        best_agent = max(performance_scores, key=performance_scores.get)
        print("\nAgent Performance Scores:")
        for agent, score in performance_scores.items():
            print(f"  {agent}: {score:.4f}")
        print(f"Selected Best Agent: {best_agent}")
        return best_agent, performance_scores

    def train_and_select(self, env, training_episodes=15, validation_episodes=2):
        training_returns = self.train_all_agents(env, training_episodes)
        validation_returns = {
            'DDPG': training_returns['DDPG'][-validation_episodes:],
            'PPO': training_returns['PPO'][-validation_episodes:],
            'A2C': training_returns['A2C'][-validation_episodes:]
        }
        best_agent_name, performance_scores = self.select_best_agent(validation_returns)
        return best_agent_name, performance_scores, training_returns

def load_and_prepare_data(filepath):

    print(f"Loading data from {filepath}")
    df = pd.read_csv("3yrs.csv")

    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    macd_indicator = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()

    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    df['CCI'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()

    df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    price_data = df['Close'].to_numpy()
    tech_indicators = {
        'MACD': df['MACD'].to_numpy(),
        'RSI': df['RSI'].to_numpy(),
        'CCI': df['CCI'].to_numpy(),
        'ADX': df['ADX'].to_numpy()
    }

    print(f"Data loaded and indicators calculated. {len(price_data)} data points.")
    return price_data, tech_indicators


def run_backtest(env, agent):
    state = env.reset()
    done = False
    history = []
    
    while not done:
        action = agent.select_action(state)
        
        if isinstance(action, tuple):  
            action = action[0]  
        if isinstance(action, (list, np.ndarray)) and len(np.shape(action)) > 0:
            action = np.array(action).flatten()[0]
        
        next_state, reward, done, info = env.step(action)
        state = next_state
        history.append(info['portfolio_value'])
        
    return pd.Series(history)

def plot_results(backtest_history, price_data, initial_balance):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    # Agent's performance
    backtest_history.plot(ax=ax, label='Ensemble Strategy', color='royalblue')

    # Buy and Hold performance
    buy_hold_value = (price_data / price_data[0]) * initial_balance
    plt.plot(buy_hold_value, label='Buy and Hold', color='gray', linestyle='--')

    ax.set_title('Trading Strategy Performance vs. Buy and Hold', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    CSV_FILE_PATH = '3yrs.csv'
    INITIAL_BALANCE = 1000000
    TRAINING_EPISODES = 50
    VALIDATION_EPISODES = 8

    price_data_np, tech_indicators = load_and_prepare_data(CSV_FILE_PATH)



    env = ImprovedNiftyTradingEnv(
        price_data=price_data_np,
        technical_indicators=tech_indicators,
        initial_balance=INITIAL_BALANCE
    )
    ensemble = ImprovedEnsembleStrategy(env.state_dim, env.action_dim)

    best_agent_name, _, _ = ensemble.train_and_select(
        env,
        training_episodes=TRAINING_EPISODES,
        validation_episodes=VALIDATION_EPISODES
    )
    best_agent = ensemble.get_agent(best_agent_name)

    print(f"\nBacktesting Best Agent: {best_agent_name}")
    backtest_env = ImprovedNiftyTradingEnv(
        price_data=price_data_np,
        technical_indicators=tech_indicators,
        initial_balance=INITIAL_BALANCE
    )
    backtest_history = run_backtest(backtest_env, best_agent)

    final_portfolio_value = backtest_history.iloc[-1]
    total_return_pct = (final_portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    
    buy_hold_final_value = (price_data_np[-1] / price_data_np[0]) * INITIAL_BALANCE
    buy_hold_return_pct = (buy_hold_final_value - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print("\nBacktest Results")
    print(f"Final Portfolio Value: ₹{final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Buy and Hold Final Value: ₹{buy_hold_final_value:,.2f}")
    print(f"Buy and Hold Return: {buy_hold_return_pct:.2f}%")

    plot_results(backtest_history, price_data_np, INITIAL_BALANCE)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_risk_metrics(portfolio_values, risk_free_rate=0.02):
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    ann_factor = 252  
    
    volatility = returns.std() * np.sqrt(ann_factor)
    sharpe_ratio = (returns.mean() * ann_factor - risk_free_rate) / volatility if volatility != 0 else np.nan
    
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(ann_factor)
    sortino_ratio = (returns.mean() * ann_factor - risk_free_rate) / downside_volatility if downside_volatility != 0 else np.nan
    
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    calmar_ratio = (returns.mean() * ann_factor - risk_free_rate) / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    return {
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio
    }

metrics = calculate_risk_metrics(backtest_history, risk_free_rate=0.02)
print("\nRisk Metrics")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")