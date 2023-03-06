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
