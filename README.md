# 2022 NTU TradeMaster

我报名了一个 quant 的竞赛，给定了15个股票的11个特征，以及 OHLC 中的 close 数据，要求使用 RL 策略计算出最优的投资组合。我的模型是参考了[Applied RL: Custom Gym environment for multi-stock RL based Algo trading](https://medium.com/@akhileshgogikar/custom-gym-environment-for-multi-stock-algo-trading-113b07dd445d#99d4)，我进行了以下修改：
- 修改了原作者的数据处理流程，使之能够work在竞赛给定的数据集上.

- 同时，增加了`test_step()`定义了测试环境中 RL Agent 的行为.

- 增加了绘图函数，将RL Agent的交易策略进行可视化，包括参考指数图、持仓分析图，以及收益率波动图.

- 在调优阶段，我还对模型的`step()`进行优化，将 reward 从收益改为夏普比率，综合考虑交易策略的收益性和波动性，力求两者的平衡.

## 1. How to Run this Repo?

本模型的实现主要依赖于 `Pytorch`, `stable baseline3`, `gym`, 并且使用 CPU 也可以进行训练.

在安装好上述包后，只需要 run `python train.py` 即可.


## 2. Trading Strategy Specification

下面对本模型的交易策略进行阐释。我们知道，对于单只股票，算法的 step 就是买入/卖出随机份额的股票。但是，对于多只股票该如何定义每步的操作呢？

首先看 `action` 是如何定义的:

```python
self.action_space = 
	spaces.Box(low=-1, high=1, shape=(num_stocks,), dtype=np.float32)

self.observation_space = 
	spaces.Box(low=-np.inf, high=np.inf, shape=(num_stocks,window_size,num_features), dtype=np.float32)
```

是一个 $num\_stocks$ 维度的向量，每个元素的取值在 $[-1, 1]$ 之间。

在给定了 $n$ 个股票的 action 后，算法会选择前 33% 个权重最高的 (当前模型最自信的) action 执行。具体的买入/卖出金额为 `(action/sum(action)) * margin`. 其中，margin 为当前持有的现金以及持有股票的市值。

> **🔍 如何计算自由多空的证券组合的投资余额？**
>
> 由于可以自由做多做空，持有的股票份数可以是负值，负值代表需要归还指定份数的股票。所以在计算账户余额时，应当扣除做空的份额，加上做多的份额。直觉上，可以这么想，如果我们有1000块，这时我们做空了1000块的股票，如果这些股票的市值跌到了 1 块钱，我们只需要买 1 块钱偿还这些股票即可，还剩余 999 块可供投资。反之，如果这些股票涨到了 2000 块，我们就亏损了 1000 块。
>
> 所以，在计算投资余额时，只需要简单的 $balance + sum(portfolio * current\_prices)$ 即可.

在执行卖出/买入操作后，更新当前的现金余额 `reserve`，同时计算收益，最后更新所有相关的类变量。
