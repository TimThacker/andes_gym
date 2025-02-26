"""
Load frequency control environment using ANDES.
This file was part of gym-power and is now part of andes_gym.
Authors:
Hantao Cui (cuihantao@gmail.com)
Yichen Zhang (whoiszyc@live.com)
Timothy Thacker (tnt.thacker@gmail.com)
Modification and redistribution of this file is subject to a collaboration agreement.
Derived source code should be made available to all authors.
"""
import os
import gym
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import andes

from gym import error, spaces, utils
from gym.utils import seeding


class AndesPrimaryFreqControlWECC(gym.Env):
    """
    Primary Frequency Control Environment using ANDES.
    This environment simulates the 179-Bus WECC System in ANDES
    with random or set load ramp disturbance. 
    Observation:
        Bus Frequency
        Bus Frequency ROCOF
        (Frequency COI)
    Action:
        Discrete action every T seconds.
        Activation of the action will adjust the `uomega0` of `TurbinGov` at action instants
        
        The governer ferquency reference is set to the action value
    Reward:
        Based on the frequency at the action instants
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Environment initialization
        """
        path = pathlib.Path(__file__).parent.absolute()
        self.path = os.path.join(path, "wecc_full_ieesgod_toggle_testcoi.xlsx")

        self.tf = 100.0     # end of simulation time
        self.tstep = 1/30  # simulation time step
        self.fixt = True   # if we do fixed step integration
        self.no_pbar = True

        # we need to let the agent to observe the disturbed trajectory before any actions taken,
        # therefore the following instant sequence is not correct: np.array([0.1, 5, 10]).
        # Instead, we will use this instant sequence: np.array([5,..., 10])
        
        # It is realistic to assume some time delay between power imbalance event and
        # identification of the frequency deviation. Thus, action application is delayed
        # by some small amount.
        
        # np.linspace(firstActionApplicationTime, lastActionApplicationTime, numberActionApplications)
        
        self.action_instants = np.linspace(1, 60, 120)

        self.N = len(self.action_instants)  # number of actions
        self.N_Gov = 29  # number of IEEEG1M models
        self.N_Bus = 29  # let it be the number of generators for now
        self.N_obs = 8   # 3 areas, COIfreq, COIrocof
        self.N_coi = 1
        self.N_area = 4

        self.action_space = spaces.Box(low=-.0001, high=.001, shape=(self.N_area,))
        self.observation_space = spaces.Box(low=-0.3, high=0.3, shape=(2*self.N_coi,))

        # This code is executed by the index of the action applications, rather than
        # the time domain simulation time step from ANDES.
        
        self.i = 0  # index of the current action

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.fig = None
        self.ax = None
        self.action_last = None

        self.t_render = None
        self.final_obs_render = None

        self.freq_print = []
        self.action_print = []
        self.reward_print = []

        # record the episode reward
        self.episode_reward = []
        # record the episode's location of disturbance
        self.episode_location = []
        
        # Record frequency of episode
        self.episode_freq = []
        self.episode_coi = []
        # Record episode of highest reward
        self.best_episode = None
        # Record best reward, this will be overwritten by anything better
        self.best_reward = -10000
        # Record frequency of best episode
        self.best_episode_freq = []
        self.coord_record = []
        self.best_coord_record = []
        self.rocof_window = []
        self.best_episode_govdata = []

        
    
    def seed(self, seed=None):
        """
        Generate the amount of load disturbance
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def initialize(self):
        """
        Initialize the andes simulation
        """
   
        self.i = 0
        self.sim_case = andes.run(self.path, no_output=True)
        
  
        
        self.sim_case.PQ.config.p2p = 1
        self.sim_case.PQ.config.p2z = 0
        self.sim_case.PQ.config.p2i = 0
        self.sim_case.PQ.config.q2q = 1
        self.sim_case.PQ.config.q2z = 0
        self.sim_case.PQ.config.q2i = 0
        self.sim_case.PQ.pq2z = 0
        self.sim_case.TDS.init()
        self.sim_case.TDS.config.criteria = 0

        # random or fixed disturbance magnitude
        #self.disturbance = 3
        self.gen_trip = 'GENROU_10'
        
        # self.disturbance = random.uniform(0.2, 0.5)
        #self.sim_case.Alter.amount.v[0] = self.disturbance
        
        # random or fixed disturbance location
        #dist_loc = 'GENROU_'+str(random.choice([2,7,10,11,12,14,19,20,21,23]))
        dist_loc = self.gen_trip
        dist_model = 'SynGen'
        #self.episode_location.append(dist_loc)
        #self.sim_case.Alter.dev.v[0] = dist_loc
        self.sim_case.Toggler.t.v[0] = 5
        self.sim_case.Toggler.dev.v[0] = dist_loc
        self.sim_case.Toggler.model.v[0] = dist_model
                                        

        # configurations
        self.sim_case.TDS.config.fixt = self.fixt

        # sensed signals
        self.w = np.array(self.sim_case.GENROU.omega.a)
        self.gov_idx = np.array(self.sim_case.IEESGOD.pout.a)
        self.coi = np.array(self.sim_case.COI.omega.a)
        self.rocof = np.array(self.sim_case.COI.rocof_y.a)
        self.dwdt = np.array(self.sim_case.BusROCOF.Wf_y.a)
        self.tg_idx = [i for i in self.sim_case.TurbineGov._idx2model.keys()]
        self.tg_idx_coi1 = ['IEESGOD_8','IEESGOD_9','IEESGOD_15','IEESGOD_16','IEESGOD_17','IEESGOD_18']
        self.tg_idx_coi2 = ['IEESGOD_1','IEESGOD_2','IEESGOD_3','IEESGOD_4','IEESGOD_7','IEESGOD_10','IEESGOD_13','IEESGOD_28','IEESGOD_29']
        self.tg_idx_coi3 = ['IEESGOD_19','IEESGOD_20','IEESGOD_21','IEESGOD_22']
        self.tg_idx_coi4 = ['IEESGOD_5','IEESGOD_6','IEESGOD_11','IEESGOD_12','IEESGOD_14','IEESGOD_23','IEESGOD_24','IEESGOD_25','IEESGOD_26','IEESGOD_27']

        self.action_last = np.zeros(self.N_Gov)

        # Step to the first action instant
        assert self.sim_to_next(), "First simulation step failed"

        self.freq_print = []
        self.action_print = []
        self.reward_print = []
        self.episode_freq = []
        self.coord_record = []
        

    def sim_to_next(self):
        """
        Simulate to the next action time instance.
        Increase the counter i each time
        """
        next_time = self.tf
        if self.i < len(self.action_instants):
            next_time = float(self.action_instants[self.i])

        self.sim_case.TDS.config.tf = next_time
        self.i += 1

        return self.sim_case.TDS.run(self.no_pbar)

    def reset(self):
        print("Env reset.")
        self.initialize()
        freq = self.sim_case.dae.y[self.coi]
        #freq = self.sim_case.dae.x[self.w]
        rocof = np.array(self.sim_case.dae.y[self.rocof]).reshape((-1, ))
        #rocof = np.array(self.sim_case.dae.y[self.dwdt]).reshape((-1, ))
        self.freq_print.append(freq[0])
        obs = np.append(freq, rocof)
        return obs

    def step(self, action):
        """
        Stepping function for RL
        """
        reward = 0.0  # reward for the current step
        done = False

        # Get the next action time in the list
        if self.i >= len(self.action_instants) - 1:  # the learning ends before the last instant
            # all actions have been taken. wrap up the simulation
            done = True

        # apply control for current step
        #coordsig=action*(1/100)
        
        #if self.i < 12:
            #windowdata = np.array(self.sim_case.dae.ts.y[:,self.rocof])
            #self.rocof_window = windowdata                         
                                 
        if self.i > 10 and self.i < 30:
            coordsig=action
            #coordsig = np.zeros(self.N_area)
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi1, value=coordsig[0], attr='v')
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi2, value=coordsig[1], attr='v')
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi3, value=coordsig[2], attr='v')
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi4, value=coordsig[3], attr='v')
            self.coord_record.append(coordsig)
        else:
            #coordsig = np.zeros(self.N_Gov)
            coordsig = np.zeros(self.N_area)
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi1, value=coordsig[0], attr='v')
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi2, value=coordsig[1], attr='v')
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi3, value=coordsig[2], attr='v')
            self.sim_case.TurbineGov.set(src='uomega0', idx=self.tg_idx_coi4, value=coordsig[3], attr='v')
            self.coord_record.append(coordsig)


        # Run andes TDS to the next time and increment self.i by 1
        sim_crashed = not self.sim_to_next()

        # get frequency and ROCOF data
        #freq = self.sim_case.dae.x[self.w]
        #coi = self.sim_case.dae.ts.y[:,self.coi]
        freq = self.sim_case.dae.y[self.coi]

        # --- Temporarily disable ROCOF ---
        rocof = np.array(self.sim_case.dae.y[self.rocof]).reshape((-1, ))
        #rocof = np.array(self.sim_case.dae.y[self.dwdt]).reshape((-1, ))
        obs = np.append(freq, rocof)

        #obs = freq

        if sim_crashed:
            reward -= 9999
            done = True

        # reward functions

        # Test: Only consider negative rewards past 5s (Gen Trip)
        
        #if self.i > 12 and not sim_crashed and done and np.max(np.abs(self.rocof_window)) > 0:
            #reward -= np.sum(np.abs(30000 * rocof ))  # the final episode
            #norm_rocof = np.divide(rocof, np.max(np.abs(self.rocof_window)))
            #reward -= 1000*np.sum(np.abs(norm_rocof))
        #elif np.max(np.abs(self.rocof_window)) > 0:
            #norm_rocof = np.divide(rocof, np.max(np.abs(self.rocof_window)))
            #reward -= 1000*np.sum(np.abs(norm_rocof))

        if not sim_crashed and done:
            #reward -= np.sum(np.abs(30000 * rocof ))  # the final episode
            reward -= 100000*np.sum(np.abs(rocof))
        else:
            reward -= 100000*np.sum(np.abs(rocof))    
        # store last action
        self.action_last = action

        # add the first frequency value to `self.freq_print`
        self.freq_print.append(freq[0])
        #self.rocof_print.append(rocof[0]) 
        #self.coi_print.append(coi[0])
        self.action_print.append(action)
        self.reward_print.append(reward)

        if done:
            self.action_total_print = []
            for i in range(len(self.action_print)):
                self.action_total_print.append(self.action_print[i])                                              
            print("Action {}".format(self.action_print))
            print("Action Total: {}".format(self.action_total_print))
            print("Freq on #0: {}".format(self.freq_print))
            #print("Rewards: {}".format(self.reward_print))
            print("Total Rewards: {}".format(sum(self.reward_print)))

            # record the episode reward
            self.episode_reward.append(sum(self.reward_print))            

            # store data for rendering. To workwround automatic resetting by VecEnv
            widx = self.w
            tmidx = self.gov_idx

            self.sim_case.dae.ts.unpack()
            xdata = self.sim_case.dae.ts.t
            ydata = self.sim_case.dae.ts.x[:, widx]
            coidata = self.sim_case.dae.ts.y[:,self.coi]
            zdata = self.sim_case.dae.ts.y[:,self.rocof]
            govdata = self.sim_case.dae.ts.y[:, tmidx]

            self.t_render = np.array(xdata)
            self.final_obs_render = np.array(ydata)
            self.final_rocof_render = np.array(zdata)
            self.final_gov_render = np.array(govdata)
            self.final_coi_render = np.array(coidata)
            
            
            if sum(self.reward_print) > self.best_reward:
                self.best_reward = sum(self.reward_print)
                #self.best_episode = self.XXXX
                self.best_episode_freq = self.final_obs_render
                self.best_coord_record = self.coord_record
                self.best_episode_rocof = self.final_rocof_render
                #if np.max(np.abs(self.rocof_window))) > 0:
                    #self.best_episode_rocof_norm = np.divide(self.final_rocof_render, np.max(np.abs(self.rocof_window)))
                self.best_episode_govdata = self.final_gov_render
                self.best_episode_coidata = self.final_coi_render
                
                                    
                                               
        return obs, reward, done, {}

    def render(self, mode='human'):

        print("Entering render...")

        if self.fig is None:
            self.fig = plt.figure(figsize=(9, 6))

            self.ax = self.fig.add_subplot(1, 1, 1)

            self.ax.set_xlim(left=0, right=np.max(self.t_render))
            self.ax.set_ylim(auto=True)
            self.ax.xaxis.set_tick_params(labelsize=16)
            self.ax.yaxis.set_tick_params(labelsize=16)
            self.ax.set_xlabel("Time [s]", fontsize=16)
            self.ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
            self.ax.ticklabel_format(useOffset=False)

            plt.ion()
        else:
            self.ax.clear()
            self.ax.set_xlim(left=0, right=np.max(self.t_render))
            self.ax.set_ylim(auto=True)
            self.ax.set_xlabel("Time [s]", fontsize=16)
            self.ax.set_ylabel("Bus Frequency [Hz]", fontsize=16)
            self.ax.ticklabel_format(useOffset=False)

        for i in range(self.N_Bus):
            self.ax.plot(self.t_render, self.best_episode_freq[:, i]*60)

        self.fig.canvas.draw()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        else:
            raise NotImplementedError

    def close(self):
        pass
