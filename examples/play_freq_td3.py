import gym
import os
import andes_gym
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3 import TD3
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
env = gym.make("AndesPrimaryFreqControlTest-v0")
# policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
#model = TD3.load('model_actnoise_0001_test.pkl')
model = TD3.load('IEEE14_Multi_Dist.zip')
save_dir = "C:/Users/tntth/andes_gym/examples/Trained_Model_Application/"

obs = env.reset()
done = False
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done is True:
        break

       
env.render()
rocof_record = pd.DataFrame(env.best_episode_rocof)
rocof_record.to_csv(save_dir + "andes_primfreq_td3_rocof_multidist_040_over.csv", index=False)
#norm_rocof_record = pd.DataFrame(env.best_episode_rocof_norm)
#norm_rocof_record.to_csv(save_dir + "andes_primfreq_td3_norm_rocof_dist_.csv", index=False)
#rocof_window = pd.DataFrame(env.rocof_window)
#rocof_window.to_csv(save_dir + "andes_primfreq_td3_rocof_window_.csv", index=False)
coord_record = pd.DataFrame(env.best_coord_record)
coord_record.to_csv(save_dir + "andes_primfreq_td3_coord_multidist_040_over.csv", index=False)
freqRec = pd.DataFrame(env.best_episode_freq)
freqRec.to_csv(save_dir + "andes_primfreq_td3_sim_multidist_040_over.csv", index=False)
gov_tm = pd.DataFrame(env.best_episode_govdata)
gov_tm.to_csv(save_dir + "andes_primfreq_td3_govdata_multidist_040_over.csv", index=False)
#totalRewards = pd.DataFrame(env.episode_reward)
#totalRewards.to_csv(save_dir + "andes_primfreq_td3_episodeRewards.csv", index=False)
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
plt.savefig(save_dir + "fig_primfreq_dynamics_multidist_040_over.pdf")
