import gym
import andes_gym
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3 import TD3
import torch
import pickle

env = gym.make("AndesPrimaryFreqControl-v0")
# policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
model = TD3.load('test_primfreq_mod.pkl')

obs = env.reset()
done = False

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done is True:
        env.render()
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
    plt.savefig("Trained_freq_dynamics.pdf")
