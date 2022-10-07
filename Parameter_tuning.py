"""
参数调优
"""
import optuna
import torch
from Config import *
from Stock_trading_environment import StockTradingEnv
import numpy as np
from stable_baselines3 import PPO

TOTAL_TIMESTEPS = int(1e5)

data = pd.read_csv(r"../Data/New_data/Stock_data.csv")

train_data = data_split(data, TRAIN_START, TRAIN_END)
validation_data = data_split(data, VALIDATION_START, VALIDATION_END)

def objective(trial):
    # parameter list
    params = {"n_step": trial.suggest_categorical("n_step", [256, 512, 1024, 2048]),
              "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
              "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]),
              "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
              "activation_fn": trial.suggest_categorical("activation_fn", ["relu", "tanh"]),
              "net_arch": trial.suggest_categorical("net_arch", [[128, 64], [64, 64, dict(pi=[32], vf=[32])]])
              }

    active_fn = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}

    policy_kwargs = dict(activation_fn=active_fn[params["activation_fn"]],
                         net_arch=params["net_arch"])

    env_train = StockTradingEnv(df=train_data)  # training environment
    env_validation = StockTradingEnv(df=validation_data)  # validation environment

    model = PPO("MlpPolicy", env_train, policy_kwargs=policy_kwargs, seed=1024,
                n_steps=params["n_step"], n_epochs=params["n_epochs"],
                batch_size=params["batch_size"], learning_rate=params["learning_rate"], verbose=0,
                device="cpu")

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    profit = np.array(backtesting_stats(model, env_validation))
    profit_ratio = profit / 1e7
    return profit_ratio.mean() / profit_ratio.std() * np.sqrt(len(profit))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=10)

best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))