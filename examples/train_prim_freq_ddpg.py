import gym
import pandas as pd
import numpy as np
import andes_gym
import os
import matplotlib.pyplot as plt
import time
import torch
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3 import DDPG

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plot_episode = True
save_dir = "C:/Users/tntth/andes_gym/examples/delay_learning_200_action_75_Primary/"

for id in range(1):
    env = gym.make('AndesPrimaryFreqControl-v0')
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])  # kwargs == keyword arguments
    model = DDPG(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, learning_starts=100)

    time_start = time.time()
    model.learn(total_timesteps=100000)  # we need to change the total steps with action numbers
    
    print("training {} completed using {}".format(id, time.time() - time_start))

    obs = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done is True:
            break
            
            
    model.save(save_dir + "andes_primfreq_ddpg_fix_{}.pkl".format(id))
    freq = pd.DataFrame(env.final_freq)
    freq.to_csv(save_dir + "andes_primfreq_ddpg_fix_{}.csv".format(id), index=False)
    freqRec = pd.DataFrame(env.best_episode_freq)
    freqRec.to_csv(save_dir + "andes_primfreq_ddpg_sim_{}.csv".format(id), index=False)
    coord_record = pd.DataFrame(env.best_coord_record)
    coord_record.to_csv(save_dir + "andes_primfreq_ddpg_coord_{}.csv".format(id), index=False)
    
    
    
    plt.rcParams.update({'font.family': 'Arial'})
    plt.figure(figsize=(9, 7))
    plt.plot(env.final_freq, color='blue', alpha=1, linewidth=2)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Frequency (Hz)", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Final frequency value per epsiode via DRL", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir + "andes_primfreq_ddpg_fix_{}.png".format(id))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(left=0, right=np.max(env.t_render))
    ax.set_ylim(auto=True)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
    ax.ticklabel_format(useOffset=False)
    for i in range(env.N_Bus):
        ax.plot(env.t_render, env.final_obs_render[:, i] * 60)
    plt.savefig("fig_primfreq_dynamics.pdf")
    for i in range(env.N_Bus):
        ax.plot(env.t_render, env.best_episode_freq[:, i] * 60)
    plt.savefig("fig_primfreq_dynamics_best.pdf")
