"""
create stock trading environment using gym

action: buy, sell and hold on
state: 63 features to represent the state of trading environment
stop criterion:
1. when finish one day trading. cur_step == day_df["step_new"].max()
2. when cur_amount + cur_shares * cur_shares_price < 0
"""

import gym
import pandas as pd
import torch
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """
    stock trading environment
    """
    def __init__(self, df, init_amount=1e7, max_volume=1e5, day=0, max_day=None, max_price=50,
                 fee_buy=0, fee_sell=0, train_mode=False):
        super(StockTradingEnv, self).__init__()

        self.df = df  # all data
        self.INIT_AMOUNT = init_amount        # initial amount(cash)
        self.MAX_VOLUME = max_volume          # maximum trading volume once time
        self.cur_amount = self.INIT_AMOUNT    # current amount(cash) in account
        self.cur_shares = 0                   # current number of holding shares
        self.bought_price = 0                 # the average buying price of holding shares
        self.MAX_PRICE = max_price            # the maximum price of shares  (default=50)
        self.cur_max_worth = self.INIT_AMOUNT # current maximum worth of account(cash+shares)
        self.total_sold_shares = 0            # total number of sold shares
        self.sold_shares_price = 0            # the average price of all sold shares
        self.total_bought_shares = 0          # total number of bought shares
        self.bought_shares_price = 0          # the average price of all bought shares

        self.cur_shares_worth = 0             # current worth of shares
        self.latest_step_worth = self.INIT_AMOUNT    # the worth in latest step

        # trading fee
        self.fee_buy = fee_buy
        self.fee_sell = fee_sell
        self.total_fee_buy = 0
        self.total_fee_sell = 0

        self.train_mode = train_mode

        assert day >= 0, print("day set wrong")
        self.INIT_DAY = day                   # model training from INIT_DAY (default=0)
        if max_day != None and max_day > self.day:
            self.MAX_DAY = max_day
        else:
            self.MAX_DAY = self.df["date_new"].max()

        # choice a day
        if self.train_mode == False:
            self.day = day
        else:   # random choice in training stage
            self.day_select_prob = np.array([0.999 ** (self.MAX_DAY - i) for i in range(self.MAX_DAY)])
            self.day_select_prob = self.day_select_prob / self.day_select_prob.sum()
            self.day = np.random.choice(list(range(self.MAX_DAY)),p=self.day_select_prob)

        self.cur_step = 0

        self.day_df = self.df[self.df["date_new"] == self.day]    # all records in self.day

        # have three actions: sell(-1), buy(+1) and hold on(0)
        # action space: [-MAX_VOLUME,……,-1,0,1,……,MAX_VOLUME]
        # action space is normalized to [-1, 1]
        # use MAX_VOLUME * action map action to action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # use 63 features to represent the state of trading environment
        self.observation_space = spaces.Box(low=0, high=1, shape=(63,), dtype=np.float32)

        self.round = 1      # data re-use round

    def _get_obs(self):
        """
        get observation according to cur_step and day
        """
        obs = self.day_df[self.day_df["step_new"] == self.cur_step].values[0]

        price = obs[-4]               # current price

        obs = list(obs[2:-4])

        # the average of historical share prices
        price_ave = self.day_df[self.day_df["step_new"] <= self.cur_step]["price"].mean()

        # the std of historical share prices
        if self.cur_step == 0:
            price_std = 0
        else:
            price_std = self.day_df[self.day_df["step_new"] <= self.cur_step]["price"].std()

        obs += [
            price / self.MAX_PRICE,
            self.cur_amount / (self.INIT_AMOUNT * 2),
            self.cur_shares / (self.MAX_VOLUME * 240),
            self.bought_price / self.MAX_PRICE,
            self.cur_max_worth / (self.INIT_AMOUNT * 2),
            self.total_sold_shares / (self.MAX_VOLUME * 120),
            self.sold_shares_price / self.MAX_PRICE,
            self.total_bought_shares / (self.MAX_VOLUME * 240),
            self.bought_shares_price / self.MAX_PRICE,
            price_ave / self.MAX_PRICE,
            price_std / self.MAX_PRICE,
            self.total_fee_buy / self.INIT_AMOUNT,
            self.total_fee_sell / self.INIT_AMOUNT
        ]

        return np.array(obs, dtype=np.float32)

    def reset(self):
        """
        reset the environment parameters
        """
        self.cur_amount = self.INIT_AMOUNT
        self.cur_shares = 0
        self.bought_price = 0
        self.cur_max_worth = self.INIT_AMOUNT
        self.cur_shares_worth = 0
        self.total_sold_shares = 0
        self.sold_shares_price = 0
        self.total_bought_shares = 0
        self.bought_shares_price = 0
        self.latest_step_worth = self.INIT_AMOUNT
        self.total_fee_buy = 0
        self.total_fee_sell = 0

        self.cur_step = 0

        self.day_df = self.df[self.df["date_new"] == self.day]

        if self.train_mode == False:
            self.day += 1
            if self.day > self.MAX_DAY:
                self.day = self.INIT_DAY
                self.round += 1
        else:
            self.day = np.random.choice(list(range(self.MAX_DAY)),p=self.day_select_prob)

        observation = self._get_obs()      # get the initial observation

        return observation

    def step(self, action):
        """
        The logic of environment
        """
        # use next step price as the actual trade price
        actual_price = self.day_df[self.day_df["step_new"] == (self.cur_step + 1)]["price"].values[0]

        if action[0] < 0:  # sell action
            actual_volume = min([int(self.MAX_VOLUME * -action[0]), self.cur_shares]) # sell volume

            if actual_volume > 0:
                self.cur_amount += actual_price * actual_volume * (1 - self.fee_sell)
                self.cur_shares -= actual_volume
                if self.cur_shares == 0:
                    self.bought_price = 0

                self.total_fee_sell += actual_price * actual_volume * self.fee_sell

                self.sold_shares_price = (self.sold_shares_price *
                                          self.total_sold_shares +
                                          actual_price * actual_volume * (1 - self.fee_sell)) /\
                                         (self.total_sold_shares + actual_volume)
                self.total_sold_shares += actual_volume

        elif action[0] > 0:  # buy action
            actual_volume = min([int(self.MAX_VOLUME * action[0]),
                                 (self.cur_amount / (1 + self.fee_buy)) // actual_price])

            if actual_volume > 0:
                self.cur_amount -= actual_price * actual_volume * (1 + self.fee_buy)
                assert self.cur_amount >= 0, "amount<0"

                self.bought_price = (self.bought_price * self.cur_shares +
                                     actual_price * actual_volume * (1 + self.fee_buy)) /\
                                    (self.cur_shares + actual_volume)
                self.cur_shares += actual_volume

                self.total_fee_buy += actual_price * actual_volume * self.fee_buy

                self.bought_shares_price = (self.bought_shares_price *
                                            self.total_bought_shares +
                                            actual_price * actual_volume * (1 + self.fee_buy)) /\
                                            (self.total_bought_shares + actual_volume)
                self.total_bought_shares += actual_volume

        else:  # hold on action
            pass

        self.cur_shares_worth = actual_price * self.cur_shares
        self.cur_max_worth = max([self.cur_max_worth,self.cur_amount + self.cur_shares_worth])

        self.cur_step += 1

        # check stop criterion
        terminated = (self.cur_step == self.day_df["step_new"].max()) or \
                     (self.cur_shares_worth + self.cur_amount) < 0

        reward = (self.cur_amount + self.cur_shares_worth - self.latest_step_worth) / 1e4
        # reward = np.log((self.cur_amount + self.cur_shares_worth) / self.latest_step_worth)  # reward is small

        self.latest_step_worth = self.cur_amount + self.cur_shares_worth

        observation = self._get_obs()

        info = {
            "step": self.cur_step,
            "current amount": self.cur_amount,
            "share worth": self.cur_shares_worth,
        }

        return observation, reward, bool(terminated), info

    def render(self, mode="human"):
        profit = self.cur_shares_worth + self.cur_amount - self.INIT_AMOUNT

        info = {
            "day": self.day,
            "current amount": self.cur_amount,
            "share worth": self.cur_shares_worth,
            "total fee": self.total_fee_buy + self.total_fee_sell,
            "profit": profit
        }
        print("-" * 50)
        for key, value in info.items():
            print("{}:{}".format(key, int(value)))

        return profit

if __name__ == '__main__':
    data = pd.read_csv(r"..\Data\New_data\Stock_data.csv")
    from Config import *
    train_data = data_split(data, TRAIN_START, TRAIN_END)

    b = StockTradingEnv(train_data)

    # check the environment
    from stable_baselines3.common.env_checker import check_env
    check_env(b, warn=True)