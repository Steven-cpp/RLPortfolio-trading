from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import StandardScaler


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

matplotlib.use("Agg")

# from stable_baselines3.common import logger


class MultiStockTradingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dfs,
        price_df,
        initial_amount,
        trade_cost,
        num_features,
        num_stocks,
        window_size,
        frame_bound,
        scalers=None,
        tech_indicator_list=[],
        reward_scaling=1e-4,
        suppresention_rate=0.66,
        representative=None
    ):
        if len(tech_indicator_list)!=0:
            num_features = len(tech_indicator_list)
        self.dfs = dfs
        self.price_df = price_df
        self.initial_amount = initial_amount
        self.margin = initial_amount
        self.portfolio = [0] * num_stocks
        self.PortfolioValue = 0
        self.reserve = initial_amount
        self.trade_cost = trade_cost
        self.state_space = num_features
        self.assets = num_stocks
        self.reward_scaling=reward_scaling
        self.tech_indicators = tech_indicator_list
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.SCALE_CONST = 0.8
        self.res_rate = [0.0] * num_stocks

        # spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_stocks,window_size,num_features), dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.price_df) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = np.zeros(self.assets)
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        self.rewards = []
        self.pvs = []
        if scalers == None:
            self.scalers = [None]*self.assets
        else:
            self.scalers =scalers
        

        self.representative = representative
        self.suppression_rate = suppresention_rate

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def process_data(self):
        signal_features = []
        for i in range(self.assets):
            df = self.dfs[i]
            start = self.frame_bound[0] - self.window_size
            end = self.frame_bound[1]
            if self.scalers[i]:
                current_scaler = self.scalers[i]
                signal_features_i = current_scaler.transform(df.loc[:, self.tech_indicators])[start:end]
            else:
                current_scaler = StandardScaler()
                signal_features_i = current_scaler.fit_transform(df.loc[:, self.tech_indicators])[start:end]
                self.scalers[i] = current_scaler
            signal_features.append(signal_features_i)

        self.prices = self.price_df.loc[:, :].to_numpy()[start:end]
        if self.representative:
            self.representative = self.price_df.loc[:, self.representative].to_numpy()[start:end]
        else:
            # 选取当天股票池中所有股票的均价作为参考指数
            self.representative = np.average(self.price_df.to_numpy(), axis=1)[start:end]
        self.signal_features = np.array(signal_features)
        self._end_tick = len(self.prices)-1
        return self.prices, self.signal_features

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._end_tick = len(self.prices)-1
        self._last_trade_tick = self._current_tick - 1
        self._position = np.zeros(self.assets)
        self._position_history = (self.window_size * [None]) + [self._position]
        self.margin = self.initial_amount
        self.portfolio = [0]*self.assets
        self.PortfolioValue = 0
        self.reserve = self.initial_amount
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.res_rate = [0.0] * self.assets
        return self._get_observation()

    def _update_profit(self, ):
        self._total_profit = (self.PortfolioValue+self.reserve)/self.initial_amount

    def testStep(self, actions):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        # 1. 处理 action, 仅保留`suppression_rate`限定个数的操作(前33%个)
        #    并且正则化使得其总和为 < 1
        delta_port = abs(actions)
        N = int(np.round(delta_port.size * self.suppression_rate))
        delta_port[np.argpartition(delta_port , kth=N)[:N]] = 0
        delta_port = delta_port / sum(delta_port)
        delta_port = np.sign(actions) * delta_port

        # 2. 执行当前操作并更新当前持仓
        self.res_rate += delta_port * self.SCALE_CONST

        observation = self._get_observation()
        return self.res_rate, self._done, observation

        

    def step(self, actions):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        # Get the current prices
        current_prices = self.prices[self._current_tick]

        # Handling cases where current price is na and avoiding buying infinite 0 cost stocks
        current_prices[np.isnan(current_prices)] = 0
        current_prices_for_division = current_prices
        current_prices_for_division[current_prices_for_division == 0] = 1e9

        # The absolute value distribution of next step portfolio
        abs_portfolio_dist = abs(actions)

        # tol = abs_portfolio_dist.mean() #+ abs_portfolio_dist.std()
        # abs_portfolio_dist[abs_portfolio_dist < tol] = 1e-9

        # At any point in time we only trade for 33% of the stocks the model is most confident about
        # the scores for the rest are suppressed

        N = int(np.round(abs_portfolio_dist.size*self.suppression_rate))
        abs_portfolio_dist[np.argpartition(abs_portfolio_dist,kth=N)[:N]] = 0
        self.margin = self.reserve + sum(self.portfolio * current_prices)

        # Normalize the portfolio positions for next step
        norm_margin_pos = (abs_portfolio_dist/sum(abs_portfolio_dist))*self.margin
        # Calulate the money in the next positions
        # next_positions = np.sign(actions) * norm_margin_pos
        next_positions = np.sign(actions) * self.initial_amount * self.SCALE_CONST # Match testStep
        # Change in money value of the positions
        change_in_positions = next_positions - self._position
        # Actions to take in the market
        actions_in_market = np.divide(change_in_positions,current_prices_for_division).astype(int)

        new_portfolio = actions_in_market + self.portfolio
        new_pv = sum(new_portfolio * current_prices)

        # 当前的持仓占比
        self.res_rate = np.divide(new_portfolio * current_prices, self.initial_amount)

        # 使用现金买入`action`指定的证券
        new_reserve = self.margin - new_pv
        profit = (new_pv + new_reserve) - (self.PortfolioValue + self.reserve)

        # Calculate the cost of each action in market
        cost = self.trade_cost * sum(abs(np.sign(actions_in_market)))
        self._position = next_positions
        self.portfolio = new_portfolio
        self.PortfolioValue = new_pv
        self.reserve = new_reserve - cost

        # Calculate the total step reward - profit made this step
        step_reward = profit - cost
        if (len(self.rewards) < 10):
            self._total_reward += self.reward_scaling * step_reward
        else:
            self._total_reward += self.reward_scaling * step_reward / np.std(self.rewards)
        self.rewards.append(self._total_reward)
        self.pvs.append(new_pv)
        self._update_profit()
        self._position = next_positions
        self._position_history.append(self._position)

        observation = self._get_observation()
        info = {'total_reward': self._total_reward,
                'total_profit': self._total_profit,
                'res_rate': self.res_rate,
                'margin': self.margin}
        self._update_history(info)

        if self.margin < 0:
            self._done = True

        return observation, step_reward, self._done, info

    def _get_observation(self):
        # 通过 current_tick 的自增来控制 obs
        return np.nan_to_num(self.signal_features[:,(self._current_tick-self.window_size+1):self._current_tick+1,:])

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.pvs)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        plt.plot(self.pvs)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
