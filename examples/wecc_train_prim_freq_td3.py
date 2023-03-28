import gym
import pandas as pd
import numpy as np
import andes_gym
import os
import matplotlib.pyplot as plt
import time
import torch
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3



##############

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, id: int, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model_{}".format(id))
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 5 episodes
              mean_reward = np.mean(y[-5:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              #if mean_reward > self.best_mean_reward and mean_reward > -900:
              if mean_reward > self.best_mean_reward and mean_reward > -1500:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


#################

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plot_episode = True
save_dir = "C:/Users/tntth/andes_gym/examples/TD3_WECC/"
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Change the range size to train a larger number of models.
for id in range(15):
    env = gym.make('AndesPrimaryFreqControlWECC-v0')
    env = Monitor(env, log_dir)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
    train_freq = (1,"episode")
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128,64])  # kwargs == keyword arguments
    model = TD3(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, action_noise=action_noise, train_freq=train_freq, batch_size=200, learning_starts=200, tensorboard_log="./td3_tensorboard_WECC/")
    callback = SaveOnBestTrainingRewardCallback(id, check_freq=300, log_dir=log_dir)
    time_start = time.time()
    model.learn(total_timesteps=50000,tb_log_name="TD3_WECC", callback=callback)  # we need to change the total steps with action numbers
    
    print("training {} completed using {}".format(id, time.time() - time_start))
    
    model.save(save_dir + "andes_primfreq_td3_model_{}.pkl".format(id))
    freqRec = pd.DataFrame(env.best_episode_freq)
    freqRec.to_csv(save_dir + "andes_primfreq_td3_sim_{}.csv".format(id), index=False)
    coord_record = pd.DataFrame(env.best_coord_record)
    coord_record.to_csv(save_dir + "andes_primfreq_td3_coord_{}.csv".format(id), index=False)
    rocof_record = pd.DataFrame(env.best_episode_rocof)
    rocof_record.to_csv(save_dir + "andes_primfreq_td3_rocof_{}.csv".format(id), index=False)
    gov_tm = pd.DataFrame(env.best_episode_govdata)
    gov_tm.to_csv(save_dir + "andes_primfreq_td3_govdata_{}.csv".format(id), index=False)
    coi= pd.DataFrame(env.best_episode_coidata)
    coi.to_csv(save_dir + "andes_primfreq_td3_coidata_{}.csv".format(id), index=False)
    #norm_rocof_record = pd.DataFrame(env.best_episode_rocof_norm)
    #norm_rocof_record.to_csv(save_dir + "andes_primfreq_td3_norm_rocof_dist_.csv", index=False)
    #rocof_window = pd.DataFrame(env.rocof_window)
    #rocof_window.to_csv(save_dir + "andes_primfreq_td3_rocof_window_.csv", index=False)
    totalRewards = pd.DataFrame(env.episode_reward)
    totalRewards.to_csv(save_dir + "andes_primfreq_td3_episodeRewards_{}.csv".format(id), index=False)

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
    plt.savefig(save_dir + "andes_primfreq_td3_rewards_{}.png".format(id))
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
    plt.savefig(save_dir+ "fig_primfreq_dynamics_best_{}.pdf".format(id))
