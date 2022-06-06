# Richard Findlay 20/03/2021
import matplotlib.pyplot as plt
import numpy as np 
import os
import pickle
from environment import floris_env
import warnings
warnings.filterwarnings("ignore")


# define bins for state spaces
ws_space = np.linspace(1, 25, 25)
wd_space = np.linspace(0, 359, 36)
yaw_space = np.linspace(0, 89, int(90 / 1.0)) # update depending on yaw rate

# function to discretize state space
def discete_state(observation, idx):

	ws, wd, _, yaw, _ = observation

	ws_bin = np.digitize(ws, ws_space)
	wd_bin = np.digitize(wd, wd_space)
	yaw_bin = np.digitize(yaw, yaw_space)

	return (ws_bin, wd_bin, yaw_bin)


# function to get best action from q-table
def max_action(Q, state, actions=[0,1,2]):

	values = np.array([Q[state, a] for a in actions])
	action = np.argmax(values)

	return action


# function to save Q-learning table
def save_table(var, filename):
	with open(filename, 'wb') as f:
		pickle.dump(var ,f)


# function to load Q-Learning table
def load_table(filename):
	with open(f'../results/{filename}', 'rb') as f:
		data = pickle.load(f)
		return data


def q_learning():

	# declare array of wind speeds
	all_ws = [8.0]

	# declare environment dictionary
	env_settings = {
		'agents': 3,
	    'wind_speeds': all_ws,
	    'number_of_turbines_x': 3,
	    'number_of_turbines_y': 1,
	    'x_spacing': 8,
	    'y_spacing': 1,
	    'run_type':'dynamic',	# 'dynamic' or 'static'
	    'render': True
	}

	env = floris_env(env_settings)

	alpha = 0.01
	gamma = 1.0
	epsilon_decay = 0.995 
	epsilon_min = 0.01
	episodes= 1000

	states = []
	for ws in range(len(ws_space)):
		for wd in range(len(wd_space)):
			for yaw in range(len(yaw_space)):
				states.append((ws, wd, yaw))

	# dictionary in dictionary for multi-agent environment
	Q_agents = {}
	Q = {}
	for agent in range(env_settings['agents']): 
		Q_agents['WT%s' %agent]= {}
		for state in states:	
			for action in [0,1,2]:
				Q_agents['WT%s' %agent][state, action] = 0.0

	tot_rew = np.zeros((episodes))

	# some lists used in training or testing
	powers = []

	# yaw increment / rate
	env.delta_gamma = 0.5


	# check for existing q-learning table
	fname = 'q_learning_table.pkl'
	if not os.path.isfile(fname):
		
		for ws in all_ws: # find optimal solution for all desired wind speeds

			epsilon = 1.0

			# intialise baseline floris model
			env.reintialise_floris(ws)

			for ep in range(episodes):
				epi_rew = 0
				
				for idx, agent in enumerate(Q_agents.keys()): 
					done = False

					obs = env.reset(idx, ws) 
					state = discete_state(obs, idx) 

					# take action for first turbine iteration, remains the same until trubine changes
					action = np.random.choice([0,1,2]) if np.random.random() < epsilon else max_action(Q_agents[agent], state)

					while not done:

						obs_, reward, done, info = env.step(action, idx, obs, ws)

						state_ = discete_state(obs_, idx)
						action_ = max_action(Q_agents[agent], state_)

						if np.all(info['locked_turbines'] == False):

							Q_agents[agent][state, action] += alpha*(reward + gamma*Q_agents[agent][state_, action_] -  Q_agents[agent][state, action])

							epi_rew += reward
						
							state = state_	

						if idx == (env_settings['agents'] - 1):
							powers.append(obs_[2])


					tot_rew[ep] += epi_rew


					print("Agent:{}\n Episode:{}\n Reward:{}\n Epsilon:{}".format(agent, ep, reward, epsilon))

				epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01

				# epsilon *= epsilon_decay
				# epsilon = max(epsilon, epsilon_min)

		# save q-learning table
		save_table(Q_agents, fname)
			
		mean_rewards = np.zeros(episodes)
		for t in range(episodes):
			mean_rewards[t] = np.mean(tot_rew[max(0, t-50):(t+1)])

		# plt.plot(mean_rewards)
		# plt.show()

		# plt.plot(timestep, powers)
		# plt.show()

		# plt.plot(powers)
		# plt.show()

	else:

		# number of test episodes
		test_episodes = [10]

		# load trained Q-learning table
		print('loading data...')
		Q_agents = load_table(fname)

		for idx, ws in enumerate(all_ws):
			env.reintialise_floris(ws)

			# test best q-learning tables and plot best result
			for ep in range(test_episodes[idx]):
				print(ep)
				epi_rew = 0
				
				for idx, agent in enumerate(Q_agents.keys()):
					done = False

					obs = env.reset(idx, ws) 
					state = discete_state(obs, idx) 

					# take action for first turbine iteration, remains the same until trubine changes
					action = max_action(Q_agents[agent], state)

					while not done:

						obs_, reward, done, info = env.step(action, idx, obs, ws)

						state_ = discete_state(obs_, idx)
						action_ = max_action(Q_agents[agent], state_)

						state = state_	



if __name__ == '__main__':

	q_learning()











