import gym
import pandas as pd
import numpy as np
import andes_gym
import os
import matplotlib.pyplot as plt
import time
import torch
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plot_episode = True
save_dir = "C:/Users/tntth/andes_gym/examples/SAC_data_ls_200_test/"

# Change the range size to train a larger number of models.
for id in range(1):
    env = gym.make('AndesPrimaryFreqControl-v0')
    n_actions = env.action_space.shape[-1]
    #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.001 * np.ones(n_actions))
    train_freq = (1,"episode")
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64, 64])  # kwargs == keyword arguments
    #model = SAC(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, action_noise=action_noise, train_freq=train_freq, learning_starts=200)
    model = SAC(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, train_freq=train_freq, learning_rate=0.0001, learning_starts=200)
    
    time_start = time.time()
    model.learn(total_timesteps=50000)  # we need to change the total steps with action numbers
    
    print("training {} completed using {}".format(id, time.time() - time_start))
    
    model.save(save_dir + "andes_primfreq_sac_model_{}.pkl".format(id))
    freqRec = pd.DataFrame(env.best_episode_freq)
    freqRec.to_csv(save_dir + "andes_primfreq_sac_sim_{}.csv".format(id), index=False)
    coord_record = pd.DataFrame(env.best_coord_record)
    coord_record.to_csv(save_dir + "andes_primfreq_sac_coord_{}.csv".format(id), index=False)
    totalRewards = pd.DataFrame(env.episode_reward)
    totalRewards.to_csv(save_dir + "andes_primfreq_sac_episodeRewards_{}.csv".format(id), index=False)

    obs = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done is True:
            break
            
    
    
    plt.rcParams.update({'font.family': 'Arial'})
    plt.figure(figsize=(9, 7))
    plt.plot(env.episode_reward, color='blue', alpha=1, linewidth=2)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Reward", fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Episode", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir + "andes_primfreq_sac_rewards_{}.png".format(id))
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
        ax.plot(env.t_render, env.best_episode_freq[:, i] * 60)
    plt.savefig(save_dir+ "fig_primfreq_dynamics_best_sac_{}.pdf".format(id))
