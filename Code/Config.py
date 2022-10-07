"""
config file
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Parameter
TRAIN_START, TRAIN_END = "2015-01-01", "2018-06-30"
VALIDATION_START, VALIDATION_END = "2018-07-01", "2018-12-31"
TEST_START, TEST_END = "2019-01-01", "2022-08-30"


def data_split(df, start, end, date_column="date"):
    """
    data split according date
    """
    df_ = df[(df[date_column]>=start) & (df[date_column]<=end)]
    df_.sort_values(by=[date_column, "step"], ascending=True, inplace=True, ignore_index=True)
    df_["date_new"] = df_[date_column].factorize()[0]
    df_["step_new"] = df_["step"]
    for date_i in df_["date_new"].unique():
        df_["step_new"][df_["date_new"] == date_i] = list(range(len(df_[df_["date_new"] == date_i])))
    return df_

# draw pic
def draw_one_day_reinner_and_price(df):
    """
    draw one day reinner and price
    """
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(df["step"], df["reinner"], label="reinner", color="b")
    ax2 = ax1.twinx()
    ln2 = ax2.plot(df["step"], df["price"], label="price", color="r")
    labs = [l.get_label() for l in ln1+ln2]
    plt.legend(ln1+ln2, labs)
    plt.show()

def draw_profit(time, profit):
    draw_data = pd.DataFrame(index=time)
    for key, value in profit.items():
        profit_i = np.cumsum(value)
        draw_data[key] = profit_i
    # print(draw_data)
    draw_data[list(profit.keys())].plot(kind="line",legend=True, ylabel="累计收益", linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# backtest
def backtesting_stats(model, env, verbose=False):
    """
    backtesting function
    """
    profit = []
    while env.round == 1:
        obs = env.reset()  # reset env
        while True:
            action = model.predict(obs)
            obs, _, terminated, info = env.step(action)
            if terminated:
                if verbose:
                    profit.append(env.render())
                else:
                    profit.append(env.cur_shares_worth + env.cur_amount - env.INIT_AMOUNT)
                break

    return profit

if __name__ == '__main__':
    pass
