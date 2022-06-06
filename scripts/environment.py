# Code implementation of NREL coference paper: https://www.nrel.gov/docs/fy20osti/75889.pdf
# Richard Findlay 20/03/2021
import floris.tools as wfct
import gym
import multiprocessing as mp
import numpy as np
import os
import pickle
from qdynplots import simulation_graphs


class floris_env(gym.Env):

	def __init__(self, env_settings):

		# instaniate turbine layout parameters
		self.n_row = env_settings['number_of_turbines_y']
		self.n_col = env_settings['number_of_turbines_x']
		self.x_space = env_settings['x_spacing']
		self.y_space = env_settings['y_spacing']
		self.run_type = env_settings['run_type']

		# intialise floris environment from json
		file_dir = os.path.dirname(os.path.abspath(__file__))
		self.fi = wfct.floris_interface.FlorisInterface(
			os.path.join(file_dir, "./_floris_params/input_params.json")
		)

		# get turbine diameter
		self.D = self.fi.floris.farm.turbines[0].rotor_diameter

		# define layout
		self.layout_x = []
		self.layout_y = []

		for i in range(self.n_row):
			for j in range(self.n_col):
				self.layout_x.append(j * self.x_space * self.D)
				self.layout_y.append(i * self.y_space * self.D)

		#instialise Floris environment 
		self.floris = self.intialise_floris()

		# establish boundaries to environment
		min_max_limits = np.array([
			[0.0, 30.0],		# wind speed [m/s]
			[0.0, 359.0],		# wind direction (degrees)
			[0.0, np.inf],		# total wind farm power
			[0, 80.0],			# yaw angle 
			[0.0, np.inf]		# reward function
		])

		self.observation_space = gym.spaces.Box(
			low=min_max_limits[:,0],
			high=min_max_limits[:,1])

		# declare action space
		self.action_space = gym.spaces.Discrete(3)

		# rate of yaw change per time step
		self.delta_gamma = 5.0

		# number of wtgs
		self.agents = 3

		# no yaw angle change
		self.no_action = 1

		# some intial parameters
		self.i = 0		# timestep
		self.ep = 0		# intialise episode ref	
		self.prev_reward = None
		self.game_over = False

		# decalre array for turbine yaws to be stored
		self.int_outer_wake_angle  = np.zeros((self.agents))
		self.intial_wake_angles = np.zeros((self.agents))
		
		# turbine yaw arrays, store previous yaws to implied dynamic env
		self.turbine_yaws = np.zeros((self.agents))
		self.pre_turbine_yaws = np.zeros((self.agents))

		# store observations between timesteps
		self.turbine_observations = {'WT' + str(wtg): 0 for wtg in range(self.agents)}
		self.turbine_refs = list(self.turbine_observations.keys())

		# Previous rewards per turbine
		self.previous_rewards = {'WT' + str(wtg): None for wtg in range(self.agents)}

		# calculate wake delay times relative to each turbine and WS
		sep_distances = {}
		outer_wake_dis = {}
		self.wake_prop_times = {}
		self.out_wake_prop_times = {}
		x_coords, y_coords = self.fi.get_turbine_layout()

		# calculate the wake propagation times
		for i in range(len(x_coords)):
			x_ref, y_ref = x_coords[i], y_coords[i]

			#calculate distances broadcasting from ref to all turbine arrays - dictionary for each turbine
			sep_distances['WT_%s' %i] = np.sqrt((x_coords - x_ref)**2 + (y_coords - y_ref)**2)

			# outer wake prop time
			average_distances = np.sum(sep_distances['WT_%s' %i]) / (self.agents)  
			outer_wake_dis['WT_%s' %i] = sep_distances['WT_%s' %i][-1] + average_distances

			# get froozen wake propopgation times
			for w in env_settings['wind_speeds']:
				self.wake_prop_times['WT%s_WS_%sm/s' %(i, w)] = sep_distances['WT_%s' %i] // w # may need to make this instance of class
				self.wake_prop_times['WT%s_WS_%sm/s' %(i, w)][:i] = 0 # make upstream turbines 0
				# calculate outer wake prop time
				self.out_wake_prop_times['WT%s_WS_%sm/s' %(i, w)] =  outer_wake_dis['WT_%s' %i] // w


		# array for saving wake delay times
		self.wake_delays = np.zeros((self.agents))
		self.out_wake_delays = -1.0 # < 0 to statisfy intial statement
		self.out_index = 0

		#previous wind speed
		self.prev_ws = None

		# set all turbines as unlocked (false)
		self.locked_turbines = np.zeros((self.agents), dtype=bool)

		# initialise turbine powers:
		self.turbine_pwrs = np.zeros((self.agents))

		# set intial previous previous turbine ref 
		self.prev_turbine = None

		# vars for animation rendering
		self.plotter_data = mp.Queue()
		self.is_plotter_active = False

		# int to follow wake propogation time for external ref in vis
		self.outer_wake_time = 0

		# set simulation to render
		if env_settings['render']:
			self.render()

		# store intial powers
		self.fi.calculate_wake()
		self.baseline_wake_pwr = self.fi.get_turbine_power()


	def reintialise_floris(self, wind_speed):
		self.floris.reinitialize_flow_field(wind_speed = wind_speed)


	def intialise_floris(self, wind_speed=8.0):

		WD = 270 # constant and only WD considered in this example

		# intialise the flow field in floris
		self.fi.reinitialize_flow_field(
			layout_array = (self.layout_x, self.layout_y), wind_direction=WD, wind_speed=wind_speed
		)

		return self.fi


	def step(self, action, idx, itnial_obs, ws):

		# refresh wake delay times for specific turbine 
		if (self.prev_turbine != idx):

			if self.run_type == 'dynamic':

				self.i += 1 # increase timestep
				# intialise wake delays
				self.wake_delays = np.copy(self.wake_prop_times['WT%s_WS_%sm/s' %(idx, ws)])

				# initlase 'settle' time
				self.settle = 100 # timesteps

			else: # env_settings['run_type'] == 'dynamic'

				self.wake_delays = np.zeros_like(self.wake_prop_times['WT%s_WS_%sm/s' %(idx, ws)])
				self.settle = 0

			# make copy of intial turbine powers
			self.intial_powers = np.copy(self.baseline_wake_pwr)

			# store intial/previous wakes before current turbine move
			self.intial_wake_angles = np.copy(self.turbine_yaws)

		# refresh outer wake delay for vertical cross visualisation
		if self.out_wake_delays <= -1 and (self.prev_turbine != idx):
			self.out_wake_delays = np.copy(self.out_wake_prop_times['WT%s_WS_%sm/s' %(self.out_index, ws)])
			self.int_outer_wake_angle = np.copy(self.intial_wake_angles)
			if self.out_index <= 1:
				self.out_index += 1
			else:
				self.out_index = 0

		# update intial previous wind speed
		if self.prev_ws is None:
			self.prev_ws = ws

		# add a time delay allowing actions to settle between different turbine actions
		if self.settle != 0:
			reward = 0
			done = False
			self.i += 1
			self.prev_turbine = idx
			if self.i > self.settle :
				self.out_wake_delays -= 1


			# return info for optmisation algorithms
			info = {
				'locked_turbines': self.locked_turbines,
				'wake_delays': self.wake_delays,
				'timestep': self.i
			}

			plotter_data_ = {
				'ts': self.i,
				'ws': self.observation[0],
				'wd': self.observation[1], 
				'turb_pwr': tuple(self.baseline_wake_pwr),
				'tot_farm_pwr': self.observation[2], 
				'turb_yaw': self.observation[3],
				'windfarm_yaws': tuple(self.turbine_yaws),
				'prev_windfarm_yaws': tuple(self.pre_turbine_yaws),
				'intial_windfarm_yaws': tuple(self.intial_wake_angles),
				'locked_turb': tuple(self.locked_turbines),
				'out_wake_delays': self.outer_wake_time,
				'intial_outer_wake':tuple(self.int_outer_wake_angle),
				'turb_ref': idx,
				'reward': 0,
				'floris': self.floris
			}

			# pass plotting data only if render is active
			if self.is_plotter_active:
				self.plotter_data.put(plotter_data_) 

			self.settle -= 1 

			return self.observation, reward, done, info

		#end simulation if observation of action out of bounds 
		self.game_over = not self.observation_space.contains(self.observation)

		# turbine yaw and reward updates only if all turbines 
		if np.all(self.locked_turbines == False):
			# take action if turbines are unlocked 
			self.turbine_yaws[idx] += ((action - 1) * self.delta_gamma)		

			# calcaulte updated wakes i.e. turbine powers 
			self.floris.calculate_wake(yaw_angles=self.turbine_yaws) 
			self.turbine_pwrs = self.floris.get_turbine_power()

			# if no turbines are yet locked immediately update power of current turbine
			if self.baseline_wake_pwr[idx] != self.turbine_pwrs[idx]:
				self.baseline_wake_pwr[idx] = self.turbine_pwrs[idx]
				self.i -= 1 # instantenous change

			# get power differences from updated action
			wake_diff = np.array(self.baseline_wake_pwr) - np.array(self.turbine_pwrs)

			# lock turbines with power change (other than the the one that underwent action)
			self.locked_turbines = np.array(wake_diff, dtype=bool)

			# make wait inducing turbine unlocked
			self.locked_turbines[idx] = False

		# check wake delays, decide if to update power of that turbine
		for t, times in enumerate(self.wake_delays):
			if times == 0 and self.baseline_wake_pwr[t] != self.turbine_pwrs[t]:
				self.baseline_wake_pwr[t] = self.turbine_pwrs[t]
				self.locked_turbines[t] = False
				self.i -= 1
			else:
				continue

		# update observations for step
		self.observation = np.array([ws, 270.0, np.sum(self.baseline_wake_pwr), self.turbine_yaws[idx], np.sum(self.baseline_wake_pwr[idx:])])

		# update current and previous yaws and powers
		(_, _, _, Y, P) = self.observation
		(_, _, _, prev_Y, prev_P) = np.array([self.prev_ws, 270.0, np.sum(self.intial_powers), self.turbine_yaws[idx], np.sum(self.intial_powers[idx:])])


		# update reward only if all turbines are unlocked
		if np.all(self.locked_turbines == False):
			P_change = P - prev_P
		else:
			P_change = 0

		# reward signals
		if P_change > 0:
			reward = 1.0
		elif abs(P_change) == 0:
			reward = 0.0
		elif P_change < 0:
			reward = -1.0

		# deter algo from choosing -ve yaws, creates instability
		if self.turbine_yaws[idx] < 0:
			reward -= 10

		done = False

		if self.game_over:
			reward = -10.0
			done = True
			self.turbine_yaws[idx] = 0

		# store some previous vars
		self.previous_rewards[self.turbine_refs[idx]] = reward
		self.prev_reward = reward
		self.prev_action = action
		self.prev_turbine = idx

		if self.prev_ws != ws:
			self.prev_ws = ws

		# update timestep		
		self.i += 1

		if np.all(self.wake_delays == 0) or np.all(self.locked_turbines == False):
			done = True

		# store lagged wake moves for visual
		for turbine in range(self.agents):
			if self.out_wake_delays == 0:
				self.pre_turbine_yaws[-1] = self.turbine_yaws[-1]
			elif turbine <= 1 and self.locked_turbines[turbine+1] == False:
				self.pre_turbine_yaws[turbine] = self.turbine_yaws[turbine]
			else:
				continue

		# update wake delay times & clip -ve to zeros
		self.wake_delays[self.wake_delays > 0] -= 1 
		self.out_wake_delays -= 1

		self.outer_wake_time = np.copy(self.out_wake_delays)

		# info for solvers
		info = {
			'locked_turbines': self.locked_turbines,
			'wake_delays': self.wake_delays,
			'timestep': self.i
		}


		# store data for graphs
		plotter_data_ = {
			'ts': self.i,
			'ws': self.observation[0],
			'wd': self.observation[1], 
			'turb_pwr': tuple(self.baseline_wake_pwr),
			'tot_farm_pwr': self.observation[2], 
			'turb_yaw': self.observation[3],
			'windfarm_yaws': tuple(self.turbine_yaws),
			'prev_windfarm_yaws': tuple(self.pre_turbine_yaws),
			'intial_windfarm_yaws': tuple(self.intial_wake_angles),
			'locked_turb': tuple(self.locked_turbines),
			'out_wake_delays': self.outer_wake_time ,
			'intial_outer_wake':tuple(self.int_outer_wake_angle),
			'turb_ref': idx,
			'reward': reward,
			'floris': self.floris
			}

		if self.is_plotter_active:
			self.plotter_data.put(plotter_data_)


		return self.observation, reward, done, info


	def reset(self, idx, ws):
		self.ep += 1
		self.prev_action = None
		self.game_over = False
		self.wake_delays = np.zeros((self.agents))

		self.observation = np.array([ws, 270.0, np.sum(self.baseline_wake_pwr), self.turbine_yaws[idx], np.sum(self.baseline_wake_pwr[idx:])])

		plotter_data_ = {
			'ts': self.i,
			'ws': self.observation[0],
			'wd': self.observation[1], 
			'turb_pwr': tuple(self.baseline_wake_pwr),
			'tot_farm_pwr': self.observation[2], 
			'turb_yaw': self.observation[3],
			'windfarm_yaws': tuple(self.turbine_yaws),
			'prev_windfarm_yaws': tuple(self.pre_turbine_yaws),
			'intial_windfarm_yaws': tuple(self.intial_wake_angles),
			'locked_turb': tuple(self.locked_turbines),
			'out_wake_delays': self.outer_wake_time,
			'intial_outer_wake': tuple(self.int_outer_wake_angle),
			'turb_ref': idx,
			'reward': 0,
			'floris': self.floris
			}	

		if self.is_plotter_active:
			self.plotter_data.put(plotter_data_)

		return self.observation


	def render(self):

		if (self.run_type == 'static'):
			print('Rendering is only available in dynamic runtime')
			raise NotImplementedError

		#some extra vars to pass to plots
		plot_vars = {'diameter': self.D,
					'agents': self.agents,
					'x_spacing': self.x_space}

		if not self.is_plotter_active:

			self.is_plotter_active = True
			self.p = mp.Process(target=simulation_graphs, args=(self.plotter_data, plot_vars,))
			self.p.start()
			# time.sleep(3)
			# self.p.join()

		elif self.is_plotter_active:
			# allow delay for mp to finalise
			time.sleep(3)





































