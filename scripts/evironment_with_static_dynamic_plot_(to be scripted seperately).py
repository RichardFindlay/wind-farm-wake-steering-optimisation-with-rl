import gym
from gym import error, spaces, utils
import floris.tools as wfct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick

import os, subprocess, time, signal, ast
import random
import logging
from datetime import datetime
from collections import deque
import multiprocessing as mp
import pickle
# import keras 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)





class Wind:

	def __init__(self, env_settings):
		self.param = env_settings['wind']
		self.dt = env_settings['timestep']
		self.t = 0.0

		assert 'mode' in self.param

		if self.param['mode'] == 'static':
			self.current_wind = self.param['ws']
			self.read = self._read_constant

		elif self.param['mode'] == 'dynamic':
			self.current_wind = self.param['speed_range'][0]
			self.target_wind = self.param['speed_range'][1]
			self.step_length = self.param['step_size']
			self.read = self._read_stepwise

		elif self.param['mode'] == 'turbulent':
			raise NotImplementedError

		else:
			raise ValueError('Unkown WindGen mode')


	def _read_constant(self, t):
		return self.param['speed']


	def _read_stepwise(self, t):
		if t > 0.0 and np.round(t, 2) % self.step_length == 0.0:
			diff_wind = np.clip(self.target_wind - self.current_wind, -1.0, 1.0)
			self.current_wind += diff_wind
		return self.current_wind 


	def reset(self):
		self.t = 0.0


# function to save Q-learning table
def save_table(var, filename):
	with open(filename, 'wb') as f:
		pickle.dump(var ,f)




# function to load Q-Learning table
def load_table(filename):
	with open(filename, 'rb') as f:
		data = pickle.load(f)
		return data





class FLORIS(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, env_settings):
		logger.debug(env_settings)
		self.dt = np.array(env_settings['timestep'],np.float32)

		# declare wind setup
		# self.wind = WndGen(env_settings)

		# instialise Floris environment 
		self.floris = self._intialise_FLORIS()


		#declare gym environment parameters
		min_max_limits = np.array([
			[0.0, 30.0],		# wind speed [m/s]
			[0.0, 359.0],		# wind direction (degrees)
			[0.0, np.inf],		# total wind farm power
			[0.0, 80.0],		# yaw angle 
			[0.0, np.inf]		# Normalised wind farm power
		])

		self.observation_space = spaces.Box(
			low=min_max_limits[:,0],
			high=min_max_limits[:,1])

		action_space_min_max = np.array([
			[-5, 5] 	# yaw (degrees / second) relative to dominate wind direction
		], np.float32) * self.dt

		self.action_space = spaces.Discrete(3)

		# self.action_space = spaces.Box(
		# 	low=action_space_min_max[:,0],
		# 	high=action_space_min_max[:,1])

		self.delta_gamma = 1.0

		self.agents = 3


		fi.calculate_wake(no_wake=True, yaw_angles=0)
		self.nowake_pwr = self.floris.get_turbine_power()
		self.nowake_pwr = np.sum(self.nowake_pwr)


		# self.no_action = np.array([0], np.float32)
		self.no_action = 1

		# self.no_action = 0

		# print(self.no_action)
		# exit()

		#simulation initial values
		self.t = 0.0 															# time
		self.t_max = env_settings['duration']									# end time
		self.i = 0																# step start
		self.i_max = int(env_settings['duration'] / env_settings['timestep'])	# total number of steps
		self.ep = 0																# intialise episode ref													
		
		self.yaw = np.array([0], np.float32) #yaw angle 

		# Rewards
		self.accum_energy = 0.0
		self.prev_reward = None
		self.game_over = False

		#declare the control varaibles
		# self.yaw_control = np.array([min_max_limits[0,0] + i for i in range(1, min_max_limits[0,1]) ])

		# Render
		self.render_antimation = False
		plt.ioff()
		self.run_timestamp = self._get_timestamp()
		#render animation
		self.plot_data = mp.Queue()
		self.active = False
		#intialise varaibles for rendering 

		# decalre array for turbine yaws to be stored
		self.turbine_yaws = np.zeros((self.agents))
		self.turbine_observations = {'WT' + str(wtg): 0 for wtg in range(self.agents)}
		self.turbine_refs = list(self.turbine_observations.keys())

		# declare previous wind speed
		self.prev_ws = 0

		# previous reward
		self.prev_reward = None

	# reintialise the model to include 
	def reintialise_floris(self, wind_speed):
		self.floris.reinitialize_flow_field(wind_speed = wind_speed)
		self.turbine_yaws = np.zeros((self.agents))
		# self.turbine_observations = {'WT' + str(wtg): 0 for wtg in range(self.agents)}
	

	def _get_timestamp(self):
		return datetime.now().strftime('%Y%m%d%H%M%S')

	def render(self):
		plt.ion()
		self.render_antimation = True


	def _intialise_FLORIS(self):

		# intialise using FlorisItnerface Object
		# instansiate the Floris Objects
		file_dir = os.path.dirname(os.path.abspath(__file__))
		fi = wfct.floris_interface.FlorisInterface(
			os.path.join(file_dir, "../example_input.json")
		)

		# declare WindSpeeds and WindDirection
		WS = [8.0]
		WD = [270]

		# define layout for enviroment
		# (1 x 3 grid, 5D spacing)
		D = fi.floris.farm.turbines[0].rotor_diameter
		n_row = 1
		n_col = 3
		x_space = 8
		y_space = 8

		layout_x = []
		layout_y = []

		for i in range(n_row):
			for j in range(n_col):
				layout_x.append(j * x_space * D)
				layout_y.append(i * y_space * D)

		turbine_num = len(layout_x)

		# Yaw Angles for each turbine
		yaws = [0, 0, 0]

		#intialise flow field 
		fi.reinitialize_flow_field(
			layout_array = (layout_x, layout_y), wind_direction=WD, wind_speed=WS
		)

		hor_plane = fi.get_hor_plane(height=fi.floris.farm.turbines[0].hub_height)

		#plot base case example
		fig, ax = plt.subplots()
		wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
		wfct.visualization.plot_turbines_with_fi(ax=ax, fi=fi)
		ax.set_title("Baseline flow for U = 8 m/s, Wind Direction = 270$^\circ$")
		plt.show()

		# # Calcaulte wake
		fi.calculate_wake(yaw_angles=yaws)

		######__FLOWFEILD_VISUALISATION__######

		# Initialize the horizontal cut
		# hor_plane = fi.get_hor_plane(height=fi.floris.farm.turbines[0].hub_height)

		# Plot and show
		# fig, ax = plt.subplots()
		# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
		# wfct.visualization.plot_turbines(
		#     ax=ax,
		#     layout_x=fi.layout_x,
		#     layout_y=fi.layout_y,
		#     yaw_angles=yaws,
		#     D=126,
		# )
		# ax.set_title("Optimised for U = 8 m/s, Wind Direction = 270$^\circ$")
		# plt.show()


		######__RESULTS__######

		# Set turbines Yaws 
		turbine_yaws = fi.get_yaw_angles()

		# Get Power from each Turbine
		turbine_pwr = fi.get_turbine_power()
		total_pwr = np.sum(turbine_pwr)

		# Get AEP
		# aep = fi.get_farm_AEP()
		# print('AEP: {aep}')

		# get_farm_power_for_yaw_angle(
		fi.show_model_parameters()

		return fi


	def _step(self, action, idx, ws):
		#take action
		# if action == 0:
		# 	action = np.array([0], np.float32)

		self.turbine_yaws[idx] += ((action -1) * self.delta_gamma)

		# update current turbine yaws
		# self.turbine_yaws[idx] = self.yaw

		print(self.turbine_yaws)

		# Simulate within environment
		# Calcaulte wake
		self.floris.calculate_wake(yaw_angles=self.turbine_yaws)
		turbine_pwr = self.floris.get_turbine_power()  
		total_pwr = np.sum(turbine_pwr[idx:]) 


		#normalise power (makes discretization easier)
		power_norm = total_pwr / self.nowake_pwr

		# hor_plane = self.floris.get_hor_plane(height=self.floris.floris.farm.turbines[0].hub_height)
		# fig, ax = plt.subplots()
		# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
		# wfct.visualization.plot_turbines_with_fi(ax=ax, fi=self.floris)
		# ax.set_title("Baseline flow for U = 8 m/s, Wind Direction = 270$^\circ$")
		# plt.show()

		# get wind speed from floris model


		observation = np.array([ws, 270.0, np.sum(turbine_pwr), self.turbine_yaws[idx], total_pwr])


		# print('****ACTION******')
		# print(action)
		# print('******YAW_ACTION***')
		# print(self.yaw)
		# print('******POWER***')
		# print(total_pwr)



		#end simulation if observation of action out of bounds 
		self.game_over = not self.observation_space.contains(observation)

		#calculate reward
		if self.prev_reward is None:
			reward = 0.0
		else:

			#get observations
			(_, _, _, Y, P) = observation
			(_, _, _, prev_Y, prev_P) = self.turbine_observations[self.turbine_refs[idx]]

			#Imply kWh in static environment
			energy = np.sum(turbine_pwr) * (self.dt / 3600.0)
			self.accum_energy += energy

			P_change = P - prev_P



			print('Change')
			print(P_change)

			delta = 1
			if P_change > 0:
				reward = 1.0
			elif abs(P_change) == 0:
				reward = 0.0
			elif P_change < 0:
				reward = -1.0


			if self.turbine_observations[self.turbine_refs[idx]][3] < 0:
				reward -= 10



			# reward = P_change - yaw_rew + alive

		# print('******REWARD*****')
		# print(reward)

		done = True

		if self.game_over:
			#penalise for fault before end of simulation
			# reward -= self.accum_energy / 1000
			reward = -10.0
			done = True
			self.yaw = 0
			self.turbine_yaws[idx] = 0

			# fig, ax = plt.subplots()
			# hor_plane = self.floris.get_hor_plane(height=self.floris.floris.farm.turbines[0].hub_height)
			# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
			# wfct.visualization.plot_turbines_with_fi(ax=ax, fi=self.floris)
			# ax.set_title("Baseline flow for U = 8 m/s, Wind Direction = 270$^\circ$")
			# plt.show()


		elif self.i == self.i_max:

			# reward += self.accum_energy / 1000 
			# reward += P_change / 1000 
			done = True


			# hor_plane = fi.get_hor_plane(height=fi.floris.farm.turbines[0].hub_height)
			# fig, ax = plt.subplots()
			# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
			# wfct.visualization.plot_turbines(
			#     ax=ax,
			#     layout_x=fi.layout_x,
			#     layout_y=fi.layout_y,
			#     yaw_angles=self.turbine_yaws,
			#     D=126,
			# )

			# ax.set_title("Optimised for U = 8 m/s, Wind Direction = 270$^\circ$")
			# plt.show()



		self.turbine_observations[self.turbine_refs[idx]] = observation
		self.prev_reward = reward
		self.prev_ws = ws

		# store turbine yaws
		# for idx in range(self.agents):
		# 	self.turbine_yaws = turbine_yaws[idx]

		#update time
		self.t += self.dt
		self.i += 1.0

		return observation, reward, done, {}



	def _reset(self, idx, ws):
		# self.anamometer.reset()
		#intial values
		self.t = 0.0
		self.i = 0.0
		self.ep += 1
		# self.yaw = 0
		self.accum_reward = 0.0

		# return observations only
		self.accum_energy = 0.0
		self.prev_reward = None
		self.game_over = False


		return self._step(self.no_action, idx, ws)[0]












#intialise using FlorisItnerface Object
# instansiate the Floris Objects
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
	os.path.join(file_dir, "../example_input.json")
)

# define layout for enviroment
# (1 x 3 grid, 5D spacing)
D = fi.floris.farm.turbines[0].rotor_diameter
n_row = 1
n_col = 3
x_space = 8
y_space = 8

layout_x = []
layout_y = []

for i in range(n_row):
	for j in range(n_col):
		layout_x.append(j * x_space * D)
		layout_y.append(i * y_space * D)


#intialise flow field 
fi.reinitialize_flow_field(
	layout_array = (layout_x, layout_y), wind_direction=[270], wind_speed=[8]
)




agents = 3

delta_gamma = 1.0

ws_space = np.linspace(1, 25, 25)
wd_space = np.linspace(0, 359, 36)
yaw_space = np.linspace(0, 89.5, int(90 / 0.5))
# yaw_space = np.linspace(0, 89, int(90 / 1.0))
pwr_space = np.linspace(0, 1, 20)



all_ws = [8.0, 12.0, 16.0, 20.0]
all_ws = [8.0]


fi.calculate_wake(no_wake=True, yaw_angles=0)
nowake_pwr = fi.get_turbine_power()
nowake_pwr = np.sum(nowake_pwr)


def discete_state(observation, idx):

	fi.calculate_wake(no_wake=True, yaw_angles=0)
	nowake_pwr = fi.get_turbine_power()
	nowake_pwr = np.sum(nowake_pwr[idx:])

	ws, wd, _, yaw, _ = observation

	ws_bin = np.digitize(ws, ws_space)
	wd_bin = np.digitize(wd, wd_space)
	yaw_bin = np.digitize(yaw, yaw_space)
	# pwr_bin = np.digitize(pwr/nowake_pwr, pwr_space)

	return (ws_bin, wd_bin, yaw_bin)


def max_action(Q, state, actions=[0,1,2]):
	values = np.array([Q[state, a] for a in actions])
	action = np.argmax(values)
	return action

def main2():

	env_settings = {
	'timestep': 1.0,
	'duration': 1.0,
	'wind': {
    'mode': 'constant',
    'speed': 8.0,}}

	env = FLORIS(env_settings)

	alpha = 0.01 
	gamma = 1.0
	epsilon_decay = 0.998 
	epsilon_min = 0.01 
	episodes= 2000 

	states = []
	for ws in range(len(ws_space)):
		for wd in range(len(wd_space)):
			for yaw in range(len(yaw_space)):
				# for pwr in range(len(pwr_space)):
				states.append((ws, wd, yaw))


	# dictionary in dictionary for multi-agent environment
	Q_agents = {}
	Q = {}
	for agent in range(agents): 
		Q_agents['WT%s' %agent]= {}
		for state in states:	
			for action in [0,1,2]:
				Q_agents['WT%s' %agent][state, action] = 0.0




	tot_rew = np.zeros((episodes))
	wf_pwr = np.zeros((episodes))

	# delta_gammas = [0.5, 2.0, 5.0]
	delta_gammas = [0.5]

	i = 0

	# check for existing q-learning table
	fname = 'q_learning_table.pkl'
	if not os.path.isfile(fname):

		powers = {}

		for delta_g in delta_gammas:

			env.delta_gamma = delta_g
			

			for ws in all_ws: # find optimal solution for all desired wind speeds
				print(ws)
				powers[str(ws)] = []
				print('*********************************')

				# intialise baseline floris model

				env.reintialise_floris(ws)

				epsilon = 1.0 

				for ep in range(episodes): # number of steps
					epi_rew = 0
					
					for idx, agent in enumerate(Q_agents.keys()): # loop over all agents (turbines) creating a unique Q-learning table for each

						done = False
						# if ep == 0:
						obs = env._reset(idx, ws)

						state = discete_state(obs, idx)

						while not done:

							action = np.random.choice([0,1,2]) if np.random.random() < epsilon else max_action(Q_agents[agent], state)

							obs_, reward, done, _ = env._step(action, idx, ws)
							state_ = discete_state(obs_, idx)

							action_ = max_action(Q_agents[agent], state_)

							Q_agents[agent][state, action] += alpha*(reward + (gamma*Q_agents[agent][state_, action_]) -  Q_agents[agent][state, action])
							state = state_

							epi_rew += reward

							if idx == 2:
								powers[str(ws)].append(obs_[2])


						tot_rew[ep] += epi_rew

			

						print("Agent:{}\n Episode:{}\n Reward:{}\n Epsilon:{}".format(agent, ep, epi_rew, epsilon))


					# epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01
					epsilon *= epsilon_decay
					epsilon = max(epsilon, epsilon_min)

				# env.turbine_yaws = [0,0,0]
		



		# save q-learning table
		save_table(Q_agents, fname)


		plt.figure(figsize=(10, 3))

		mean_rewards = np.zeros(episodes)
		for t in range(episodes):
			mean_rewards[t] = np.mean(tot_rew[max(0, t-25):(t+1)])


		plt.style.use(['seaborn-whitegrid'])

		with open('../dynamic_powers.pkl', 'rb') as f:
			data = pickle.load(f)



		mean_powers = {}
		for idx, ws in enumerate(all_ws):
			mean_powers[str(ws)] = np.zeros(len(powers[str(ws)]))
			for p in range(len(powers[str(ws)])):
				mean_powers[str(ws)][p] = np.mean(powers[str(ws)][max(0, p-5):(p+1)])

		ax = plt.subplot()

		colors = ['dodgerblue', 'tomato', 'green', 'black']
		for idx, ws in enumerate(all_ws):
			plt.plot(np.array(powers[str(ws)]) / 1000000.0, color=colors[idx], label=r'Static $\bf{|}$ $\Delta\gamma$ = 0.5', alpha=0.5, zorder=5)
			plt.plot(np.array(mean_powers[str(ws)]) / 1000000.0, color=colors[idx], lw=0.5)

		
		ax.grid(True, alpha=0.6, which="both")
		ax.spines['bottom'].set_color('black')  
		ax.spines['left'].set_color('black')
		ax.tick_params(direction="out", length=2.0)

		ax.tick_params(axis='y', labelsize= 8)
		ax.tick_params(axis='x', labelsize= 8)
		ax.set_ylabel('Power (MW)', fontsize=9, style='italic', weight='bold')
		ax.set_xlabel('Iteration', fontsize=9, style='italic', weight='bold')
		ax.grid(alpha=0.3)

		sec_ax = ax.twiny()
		sec_ax.grid(False)

		sec_ax.plot(np.array(data)/ 1000000.0, lw="0.75", label=r'Quasi-dynamic $\bf{|}$ $\Delta\gamma$ = 5.0', color='tomato', alpha=0.8)

		# mean_powers2 = np.zeros(len(data))
		# for p in range(len(data)):
		# 	mean_powers2[p] = np.mean(data[max(0, p-1000):(p+1)])

		# sec_ax.plot(mean_powers2/ 1000000.0, lw="0.5", color='tomato', alpha=1.0, zorder=1) 

		ax.xaxis.set_major_locator(mtick.LinearLocator(8))
		sec_ax.xaxis.set_major_locator(mtick.LinearLocator(8))
		sec_ax.tick_params(axis='y', labelsize= 8)
		sec_ax.tick_params(axis='x', labelsize= 8)
		sec_ax.set_xlabel('Timestep', fontsize=9, style='italic', weight='bold')

		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['right'].set_visible(False)
		sec_ax.spines['top'].set_color('black')
		sec_ax.spines['bottom'].set_color('black')
		sec_ax.spines['left'].set_color('black')
		sec_ax.spines['right'].set_color('black')

		ax.set_xlim([0, 2000])
		sec_ax.set_xlim([0, 31100])

		ax.tick_params(direction="out", length=2.0)
		sec_ax.tick_params(direction="out", length=2.0)

		ax.set_xticks(np.arange(0, 2001, 200))
		sec_ax.set_xticks(np.arange(0, 31101, 3110))

		handle1, label1 = ax.get_legend_handles_labels()
		handle2, label2 = sec_ax.get_legend_handles_labels()


		leg = sec_ax.legend(handle1+handle2, label1+label2, loc="lower right", fontsize=7, frameon=True)
		leg.set_zorder(5)

		for line in leg.get_lines():
			line.set_linewidth(1.5)

		frame = leg.get_frame()
		frame.set_facecolor('white')
		frame.set_edgecolor('white')

		plt.savefig('foo2.png', bbox_inches='tight', transparent=True)

		plt.show()

		



		# baseline power
		fi.calculate_wake(yaw_angles=0)
		nowake_pwr = fi.get_turbine_power()
		nowake_pwr = np.sum(nowake_pwr)
		print('Baseline: %s' %nowake_pwr)

		fi.calculate_wake(yaw_angles=[-58.5, -58.5, -58.5])
		op_pwr = fi.get_turbine_power()
		op_pwr = np.sum(op_pwr)
		print('Optimised: %s' %op_pwr)


	else:


		# load trained Q-learning table
		print('loading data...')
		Q_agents = load_table(fname)

		# print(Q_agents['WT2'])
		# ((0, 0, 33), 1)
		# print(list(Q_agents['WT0'].keys())[90:100])
		# print(Q_agents['WT1'][(8, 27, 177), 2])
		# print(type(Q_agents['WT1'][(8, 27, 172), 0]))
		print(Q_agents.keys())

		# exit()

		# # test environment settings
		test_episodes = 50
		best_yaws = np.zeros((agents))

		# test_env_settings = {
		# 	'timestep': 1.0,
		# 	'duration': 1.0,
		# 	'wind': {
		#     'mode': 'constant',
		#     'speed': 8.0,}}

		# env = FLORIS(test_env_settings)
		epsilon  = 0.01

		powers = []

		for ws in all_ws:

			print('*********************************')
			print(ws)

			# intialise baseline floris model
			env.reintialise_floris(ws)

			# test best q-learning tables and plot best result
			for episode in range(test_episodes):

				for idx, agent in enumerate(Q_agents.keys()):
					print(agent)

					done = False
					# if episode == 0:
					obs = env._reset(idx, ws)
					state = discete_state(obs, idx)

					while not done:
						
						action = max_action(Q_agents[agent], state)
						print(action)

						obs_, reward, done, _ = env._step(action, idx, ws)
						state_ = discete_state(obs_, idx)

						state = state_

						# if action == 1:
						# 	done = True

						powers.append(obs_[2])



			plt.plot(powers)
			plt.show()



					# best_yaws[idx] = obs_[2]

			

		# fig, ax = plt.subplots()
		# hor_plane = fi.get_hor_plane(height=fi.floris.farm.turbines[0].hub_height)
		# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
		# wfct.visualization.plot_turbines(ax=ax, layout_x=fi.layout_x, layout_y=fi.layout_y, D=126, yaw_angles=best_yaws)
		# ax.set_title("Optmisied_Result")
		# plt.show()



main2()










