from tabnanny import verbose
import talib
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import glob
import pandas as pd

from multi_stock_trading_env import MultiStockTradingEnv
from stable_baselines3 import PPO, A2C, SAC, DDPG
from custom_rl_policy import CustomActorCriticPolicy, CustomNetwork
CUDA_LAUNCH_BLOCKING=1 

name = "MultiStockTrader_stable"
indicators = ['feature_%d' %i for i in range(11)]
cols_per_asset = 11
num_assets = 15
names = ['stock_%d' %i for i in range(num_assets)]

"""
Read history indicators from `dir`

Parameters
----------
* fname : str
    证券历史技术指标所在的文件

Example Usage
-------------
>>> readHistory('public_data/train.csv', isTrain=True)

"""
def readHistory(fname, isTrain):
    df = pd.read_csv(fname)
    df_list = []
    price_df = pd.DataFrame()

    dfs = [x for _, x in df.groupby('tic')]

    for i in range(num_assets):
        # Cover all the features and close price
        df = dfs[i].iloc[:, 1: 3+cols_per_asset]
        df = df.drop(columns=['date']).reset_index()
        if not isTrain:
            price_df[names[i]] = df[indicators[0]]
        else:
            price_df[names[i]] = df['close']
        df_list.append(df)
    return df_list, price_df, num_assets, cols_per_asset - 1 

# df_list, price_df, num_assets, cols_per_asset = readHistory('public_data/train.csv', isTrain=True)

def trainRL():
    env = MultiStockTradingEnv(df_list,
            price_df,
            num_stocks=num_assets,
            initial_amount=100000,
            trade_cost=0,
            num_features=cols_per_asset,
            window_size=3,
            frame_bound = (3, len(price_df) - 800),
            tech_indicator_list=indicators)
    env.process_data()
    # , batch_size=256
    model = A2C(CustomActorCriticPolicy, env, verbose=2,tensorboard_log='tb_logs')
    # model = PPO(CustomNetwork, env, verbose=2,tensorboard_log='tb_logs', batch_size=256)
    # model = PPO('MlpPolicy', env, verbose=2,tensorboard_log='tb_logs', batch_size=256)
    # model = PPO(GATActorCriticPolicy, env, verbose=2,tensorboard_log='tb_logs', batch_size=16)

    model.learn(total_timesteps=8000)
    plt.figure(figsize=(16, 6))
    model.save("saved_models/" + name)
    scalers = env.scalers
    print(len(scalers))

if __name__ == "__main__":
    df_list, price_df, num_assets, cols_per_asset = readHistory('public_data/test_input_2.csv', isTrain=False)
    # trainRL()
    env = MultiStockTradingEnv(df_list,
            price_df,
            num_stocks=num_assets,
            initial_amount=100000,
            trade_cost=0,
            num_features=cols_per_asset,
            window_size=3,
            # frame_bound = (len(price_df) - 800, len(price_df)),
            frame_bound = (5, len(price_df)),
            tech_indicator_list=indicators)
    model = A2C.load("saved_models/" + name)
    env.process_data()
    obs = env.reset()
    count = 0
    total_rewards = 0
    infer_rewards = []
    svps = []
    port_ratios = []

    # 模型测试
    while True:
        action, _states = model.predict(obs, deterministic=True)
        count += 1
        port_ratio, done, obs = env.testStep(action)
        temp = list(port_ratio)
        port_ratios.append(temp)
        if done:
            break
    port_ratios = np.array(port_ratios)
    np.save('res/action2', port_ratios)


"""
    obs = env.reset()
    port_ratios = []
    margins = []
    last_er = 0
    while True: 
        # obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs, deterministic=True)
        count += 1
        obs, rewards, done, info = env.step(action)
        # print(action, rewards)
        # 每只股票的市值
        sv = env.portfolio * env.prices[env._current_tick]
        # 持仓占比
        svp = abs(sv) / env.initial_amount
        temp = list(info['res_rate'])
        port_ratios.append(temp)
        margins.append(info['margin'])
        # (day, 股票名)
        svps.append(svp)
        total_rewards += rewards
        infer_rewards.append(rewards)
        last_er = info['total_profit']
        if done:
            print("info", count, info)
            break

    print("Total profit: %.3f" % sum(infer_rewards))
    flucation = (max(margins) - min(margins)) / env.initial_amount
    print('Fluctuation : %.3f' % flucation)
    print('ER - Fluc = %.3f' % (2 * (last_er - 1) - flucation))
    port_ratios = np.array(port_ratios)
    np.save("res/train_res", port_ratios)

    infer_steps = price_df.index[len(price_df)-len(infer_rewards):len(price_df)]
    infer_rewards = np.cumsum(np.array(infer_rewards))
    sensex_values = env.representative[-len(infer_steps):]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
    svps = np.array(svps)
    ax1.set_title('Average Index')
    ax1.plot(infer_steps, sensex_values, label='Index')
    ax2.set_title('Infer Rewards')
    ax2.plot(infer_steps, infer_rewards, color="red", label='Profit')
    ax3.set_title('Overall Position')
    ax3.plot(infer_steps, np.sum(port_ratios, axis=1))
    # ax2.legend()
    # ax2.plot(infer_steps, svps[:, 1], label=names[1])
    # ax2.plot(infer_steps, svps[:, 2], label=names[2])
    fig.savefig('inferRewards_multi.jpg')

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.title('Reward Variation')
    plt.xlabel('Day')
    plt.ylabel('Reward')
    plt.plot(infer_steps, env.rewards)
    plt.savefig('reward.jpg')
"""

