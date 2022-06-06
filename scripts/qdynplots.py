# Richard Findlay 20/03/2021
import floris.tools as wfct
import numpy as np
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import ticker as mtick
from matplotlib.ticker import FormatStrFormatter
import multiprocessing as mp
from multiprocessing import Process, Manager


def simulation_graphs(graph_data, plot_vars):

	plt.style.use(['seaborn-whitegrid'])

	plt.ion() #activate interactive plot

	# store some pertinent vars
	diameter = plot_vars['diameter']
	agents = plot_vars['agents']
	x_spacing = plot_vars['x_spacing']


	# Declare lists for graph vars
	time_all = []
	windfarm_power_all = []
	reward_all =[]
	ws_all = []
	first_turbine_powers_all = []

	# some vars relative to each agent, hence use dictionary of lists
	# declare dictionaries
	turbine_yaws_all = {}
	turbine_powers_all = {}
	for agent in range(agents):
		turbine_yaws_all['turbine_%s' %agent] = []
		
	for index in range(1,3):
		turbine_powers_all['turbine_%s' %index] = []

	# figure setup
	fig = plt.figure(figsize=(8, 12), constrained_layout=False)
	gs = gridspec.GridSpec(ncols=50, nrows=230)

	# subplot layouts
	row_start = 43
	col_width = 50

	trb_plan = fig.add_subplot(gs[0:row_start, 0:col_width])

	# create wake profile sub plots, depending on number of agents
	wake_profiles = {}
	cross_sections = {}
	row_spacing = row_start
	col_spacing = 0

	for idx in range(1, agents+1):

		# create dictionary of wake profiles  
		wake_profiles['wake_profile_%s' %idx] = fig.add_subplot(gs[(row_spacing + 5):(row_spacing + 40), (col_spacing + (idx-1)):(col_spacing + (idx-1) + 16)]) 
		wake_profiles['wake_profile_%s' %idx].set_facecolor('#b40826') # compensate for reduce plot size at higher WS

		# update col_spcaing to a fixed third of the window
		col_spacing = col_spacing + 16

		# for each for of three increase row_spacing
		if (idx * agents) == 0:
			row_spacing += 30 
			col_spacing = 0 

	# increase buffer between graphs
	row_spacing += 8

	# plot line graphs for power, reward etc.
	wf_pwr = fig.add_subplot(gs[row_spacing+45:row_spacing+80, 0:50])
	trb_yaws = fig.add_subplot(gs[row_spacing+88:row_spacing+123, 0:50])
	trb_pwrs = fig.add_subplot(gs[row_spacing+131:row_spacing+166, 0:50])

	# remove borders from 
	# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

	# function to update existing y-labels from watts to mega-watts
	def pwrfmt(x, pos): 
		s = '{:0.2f}'.format(x / 1000000.0) # watts to MW
		return s

	yfmt = mtick.FuncFormatter(pwrfmt)

	#######_First_Line_Plot_#######################################################################################
	# Left Axis -> Wind Farm power | Right Axis -> Wind Speed
	wf_pwr.grid(True, alpha=0.6, which="both")

	wf_pwr_line, = wf_pwr.plot([],[], color='dodgerblue', lw="1.0", label='Total Power', zorder=2, clip_on=False)
	wf_pwr.set_title('Wind Farm Power (MW) & Wind Speed (m/s)', fontsize=8, fontweight='bold')
	wf_pwr.set_xticklabels([])
	wf_pwr.tick_params(axis='y', labelsize= 7)
	wf_pwr.set_ylabel('Power (MW)', fontsize=8, style='italic')

	# intialise plot for second axis
	sec_ax_wfpwr = wf_pwr.twinx()
	sec_ax_wfpwr.grid(False)
	
	wf_pwr_line_sec, = sec_ax_wfpwr.plot([],[], color='#fb743e', lw="1.0", label='Wind Speed', clip_on=False, zorder=2)
	sec_ax_wfpwr.tick_params(axis='y', labelsize= 7)
	sec_ax_wfpwr.set_ylabel('Wind Speed (m/s)', fontsize=8, style='italic')
	# clean up splines - both axes
	wf_pwr.spines['top'].set_visible(False)
	wf_pwr.spines['bottom'].set_visible(False)
	wf_pwr.spines['left'].set_visible(False)
	wf_pwr.spines['right'].set_visible(False)
	sec_ax_wfpwr.spines['top'].set_color('black')
	sec_ax_wfpwr.spines['bottom'].set_color('black')
	sec_ax_wfpwr.spines['left'].set_color('black')
	sec_ax_wfpwr.spines['right'].set_color('black')
	# line-up gridlines 
	wf_pwr.yaxis.set_major_locator(mtick.LinearLocator(6))
	sec_ax_wfpwr.yaxis.set_major_locator(mtick.LinearLocator(6))
	# add minor ticks and locate
	wf_pwr.minorticks_on()
	wf_pwr.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
	wf_pwr.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
	# add tick marks
	wf_pwr.tick_params(direction="out", length=2.0)
	sec_ax_wfpwr.tick_params(direction="out", length=2.0)
	# axis limits 
	wf_pwr.set_ylim([3055949, 3377628])
	wf_pwr.set_xlim([0, 1500])	
	sec_ax_wfpwr.set_ylim([0, 10])
	# convert watts to mega-watts
	wf_pwr.yaxis.set_major_formatter(yfmt)
	sec_ax_wfpwr.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	# plot single legend for multiple axes
	handle1, label1 = wf_pwr.get_legend_handles_labels()
	handle2, label2 = sec_ax_wfpwr.get_legend_handles_labels()
	leg = wf_pwr.legend(handle1+handle2, label1+label2, loc="upper right", fontsize=7, frameon=True)
	leg.set_zorder(5)
	# line width for ledgend
	for line in leg.get_lines():
	    line.set_linewidth(1.5)
	# legend background and frame color
	frame = leg.get_frame()
	frame.set_facecolor('white')
	frame.set_edgecolor('white')
 
	#######_Second_Line_Plot_#######################################################################################
	# Left Axis -> Turbine Yaws | Right Axis -> Reward
	trb_yaws.grid(True, alpha=0.6, which="both")

	# plot first axis
	color = ['dodgerblue', '#28527a', '#fb743e']
	turbine_refs = ['WTG_1', 'WTG_2', 'WTG_3']
	turb_yaw_lines = []
	for agent in range(agents): 
		trb_yaw_line, = trb_yaws.plot([],[], color=color[agent], lw="1.0", label=turbine_refs[agent], zorder=2, clip_on=False)
		turb_yaw_lines.append(trb_yaw_line)
	
	trb_yaws.set_title('Turbine Yaw Angles ($^\circ$) & Reward', fontsize=8, fontweight='bold', zorder=10)
	trb_yaws.set_xticklabels([])
	trb_yaws.tick_params(axis='y', labelsize= 7)
	trb_yaws.set_ylabel('Yaw Angle ($^\circ$)', fontsize=8, style='italic')

	# intialise plot for second axis
	sec_ax_trb_yaws = trb_yaws.twinx()
	sec_ax_trb_yaws.grid(False)
	trb_yaws_sec, = sec_ax_trb_yaws.plot([],[], color='tomato', lw="1.0", label='Reward', zorder=2, clip_on=False, alpha=0.6, linestyle='dashed')
	sec_ax_trb_yaws.tick_params(axis='y', labelsize=7)
	sec_ax_trb_yaws.set_ylabel('Reward', fontsize=8, style='italic')
	# clean up splines - both axes
	trb_yaws.spines['top'].set_visible(False)
	trb_yaws.spines['bottom'].set_visible(False)
	trb_yaws.spines['left'].set_visible(False)
	trb_yaws.spines['right'].set_visible(False)
	sec_ax_trb_yaws.spines['top'].set_color('black')
	sec_ax_trb_yaws.spines['bottom'].set_color('black')
	sec_ax_trb_yaws.spines['left'].set_color('black')
	sec_ax_trb_yaws.spines['right'].set_color('black')
	# line-up gridlines 
	trb_yaws.yaxis.set_major_locator(mtick.LinearLocator(6))
	sec_ax_trb_yaws.yaxis.set_major_locator(mtick.LinearLocator(6))
	# add minor ticks and locate
	trb_yaws.tick_params(direction="out", length=2.0)
	sec_ax_trb_yaws.tick_params(direction="out", length=2.0)
	trb_yaws.minorticks_on()
	trb_yaws.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
	trb_yaws.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
	# axis limits
	trb_yaws.set_ylim([-1, 10])
	trb_yaws.set_xlim([0, 1500])
	sec_ax_trb_yaws.set_ylim([-5, 5])
	# format labels	
	trb_yaws.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	sec_ax_trb_yaws.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	# plot single legend for multiple axes
	handle1, label1 = trb_yaws.get_legend_handles_labels()
	handle2, label2 = sec_ax_trb_yaws.get_legend_handles_labels()
	leg = sec_ax_trb_yaws.legend(handle1+handle2, label1+label2, loc="upper right", fontsize=7, frameon=True)
	leg.set_zorder(5)
	# line width for ledgend
	for line in leg.get_lines():
	    line.set_linewidth(1.5)
	# legend background and frame color
	frame = leg.get_frame()
	frame.set_facecolor('white')
	frame.set_edgecolor('white')

	#######_Third_Line_Plot_#######################################################################################
	# Left Axis -> Individual Turbine Powers (WTG1) | Right Axis -> Individual Turbine Powers (WTG2 & 3)
	trb_pwrs.grid(True, alpha=0.6, which="both")

	trb_pwrs_line, = trb_pwrs.plot([],[], color='dodgerblue', lw="1.0", label='WTG_1', zorder=2, clip_on=False)
	trb_pwrs.set_title('Individual Turbine Power (MW)', fontsize=8, fontweight='bold')
	trb_pwrs.set_xlabel('Timestep', fontsize=8, style='italic')
	trb_pwrs.tick_params(axis='y', labelsize= 7)
	trb_pwrs.tick_params(axis='x', labelsize= 8)
	trb_pwrs.set_ylabel('Power (MW) - WTG_1', fontsize=8, style='italic')

	# instaniate and plot second axis
	sec_ax_trb_pwrs = trb_pwrs.twinx()
	sec_ax_trb_pwrs.grid(False)

	color = ['#28527a', '#fb743e']
	turbine_refs = ['WTG_2', 'WTG_3']
	sec_trb_pwrs_lines = []
	for agent in range(agents-1): 
		trb_pwrs_sec, = sec_ax_trb_pwrs.plot([],[], color=color[agent], lw="1.0", label=turbine_refs[agent], zorder=2, clip_on=False)
		sec_trb_pwrs_lines.append(trb_pwrs_sec)

	# general format of second access
	
	sec_ax_trb_pwrs.tick_params(axis='y', labelsize= 7)
	sec_ax_trb_pwrs.set_ylabel('Power (MW) - WTGs_2 & 3', fontsize=8, style='italic', zorder=10)
	# clean up splines
	trb_pwrs.spines['top'].set_visible(False)
	trb_pwrs.spines['bottom'].set_visible(False)
	trb_pwrs.spines['left'].set_visible(False)
	trb_pwrs.spines['right'].set_visible(False)
	sec_ax_trb_pwrs.spines['top'].set_color('black')
	sec_ax_trb_pwrs.spines['bottom'].set_color('black')
	sec_ax_trb_pwrs.spines['left'].set_color('black')
	sec_ax_trb_pwrs.spines['right'].set_color('black')
	# line-up gridlines 
	trb_pwrs.yaxis.set_major_locator(mtick.LinearLocator(6))
	sec_ax_trb_pwrs.yaxis.set_major_locator(mtick.LinearLocator(6))
	# add minor ticks and locate
	trb_pwrs.tick_params(direction="out", length=2.0)
	sec_ax_trb_pwrs.tick_params(direction="out", length=2.0)
	trb_pwrs.minorticks_on()
	trb_pwrs.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
	trb_pwrs.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
	# axis limits
	trb_pwrs.set_ylim([1542785, 1712322])
	sec_ax_trb_pwrs.set_ylim([676376, 849917])
	trb_pwrs.set_xlim([0, 1500])
	# convert watts to mega-watts
	trb_pwrs.yaxis.set_major_formatter(yfmt)
	sec_ax_trb_pwrs.yaxis.set_major_formatter(yfmt)
	# plot single legend for multiple axes
	handle1, label1 = trb_pwrs.get_legend_handles_labels()
	handle2, label2 = sec_ax_trb_pwrs.get_legend_handles_labels()
	leg = sec_ax_trb_pwrs.legend(handle1+handle2, label1+label2, loc="upper right", fontsize=7, frameon=True)
	leg.set_zorder(5)
	# line width for ledgend
	for line in leg.get_lines():
	    line.set_linewidth(1.5)
	# legend background and frame color
	frame = leg.get_frame()
	frame.set_facecolor('white')
	frame.set_edgecolor('white')


	# define update function for animation
	def update(i):

		# get data from queue
		data = graph_data.get()

		# get current floris obj from queue
		fi = data['floris']

		# collate data from queue
		time_all.append(data['ts'])
		ws_all.append(data['ws'])
		windfarm_power_all.append(data['tot_farm_pwr'])

		# get turbine yaw data
		current_wtg_ref = data['turb_ref']

		for agent in range(agents):
			if current_wtg_ref == agent:
				turbine_yaws_all['turbine_%s' %agent].append(data['turb_yaw'])
			elif not turbine_yaws_all['turbine_%s' %agent]:
				turbine_yaws_all['turbine_%s' %agent].append(0)
			else:
				turbine_yaws_all['turbine_%s' %agent].append(turbine_yaws_all['turbine_%s' %agent][-1])

		# turbine power for primary axis
		first_turbine_powers_all.append(data['turb_pwr'][0])

		# turbine powers for secondary axis
		for agent in range(1,3):
			turbine_powers_all['turbine_%s' %agent].append(data['turb_pwr'][agent])

		# cumaltive reward
		if len(reward_all) > 0: 
			cumlative_reward = reward_all[-1] + data['reward']
		else:
			cumlative_reward = 0

		reward_all.append(cumlative_reward)


		# get reference for x-axis scales
		_ , xlim = trb_yaws.get_xlim()
		if data['ts'] >= xlim:
			# change x-axis range when data out of range
			wf_pwr.set_xlim(0, xlim * 2)
			wf_pwr.figure.canvas.draw()
			trb_yaws.set_xlim(0, xlim * 2)
			trb_yaws.figure.canvas.draw()
			trb_pwrs.set_xlim(0, xlim * 2)
			trb_pwrs.figure.canvas.draw()

		# First line graph re-scaling (left axis -> wind farm power)
		ylimMin_PWR, ylimMax_PWR = wf_pwr.get_ylim()
		if data['tot_farm_pwr'] >= (ylimMax_PWR * 0.99):
			wf_pwr.set_ylim(ylimMin_PWR, data['tot_farm_pwr'] + (0.01*data['tot_farm_pwr']))
		elif data['tot_farm_pwr'] <= ylimMin_PWR:
			wf_pwr.set_ylim((data['tot_farm_pwr'] - (0.01*data['tot_farm_pwr'])), ylimMax_PWR)

		# First line graph re-scaling (right axis -> wind speed)
		ylimMin_WS, ylimMax_WS = sec_ax_wfpwr.get_ylim()
		if data['ws'] >= (ylimMax_WS * 0.99):
			sec_ax_wfpwr.set_ylim(ylimMin_WS, ylimMax_WS*2)

		# Second line graph re-scaling (left axis -> turbine yaws)
		ylimMin_turbYAW, ylimMax_turbYAW = trb_yaws.get_ylim()
		if data['turb_yaw'] >= (ylimMax_turbYAW * 0.99):
			trb_yaws.set_ylim(ylimMin_turbYAW, ylimMax_turbYAW+10)

		# Second line graph re-scaling (right axis -> reward)
		ylimMin_REW, ylimMax_REW = sec_ax_trb_yaws.get_ylim()
		if cumlative_reward >= (ylimMax_REW * 0.99):
			sec_ax_trb_yaws.set_ylim(ylimMin_REW, ylimMax_REW+5)
		elif cumlative_reward <= (ylimMin_REW * 1.01):
			sec_ax_trb_yaws.set_ylim(ylimMin_REW-5, ylimMax_REW)

		# Third line graph re-scaling (left axis -> turbine 1 power)
		ylimMin_turbPWR, ylimMax_turbPWR = trb_pwrs.get_ylim()
		if data['turb_pwr'][0] >= (ylimMax_turbPWR * 0.99):
			trb_pwrs.set_ylim(ylimMin_turbPWR, data['turb_pwr'][0] + (0.07 * data['turb_pwr'][0]))
		elif data['turb_pwr'][0] <= (ylimMin_turbPWR):
			trb_pwrs.set_ylim(data['turb_pwr'][0] - (0.07 * data['turb_pwr'][0]), ylimMax_turbPWR)

		# Third line graph re-scaling (right axis -> turbine 2 & 3 power)
		ylimMin_turbPWR, ylimMax_turbPWR = sec_ax_trb_pwrs.get_ylim()
		if (data['turb_pwr'][1] >= ylimMax_turbPWR *0.99) or (data['turb_pwr'][2] >= ylimMax_turbPWR *0.99):
			max_val = np.max(data['turb_pwr'][1:])
			sec_ax_trb_pwrs.set_ylim(ylimMin_turbPWR, max_val + (0.05 * max_val))
		elif (data['turb_pwr'][1] <= ylimMin_turbPWR*1.01) or (data['turb_pwr'][2] <= ylimMin_turbPWR*1.01):
			min_val = np.max(data['turb_pwr'][1:])
			sec_ax_trb_pwrs.set_ylim(min_val + (0.07 * min_val), ylimMax_turbPWR)

		# clear previous cross sectional plots
		trb_plan.cla()

		for idx in range(1, agents+1):
			wake_profiles['wake_profile_%s' %idx].cla() 

		# Update plan view cross section 
		wake_calcs = fi.calculate_wake(yaw_angles=list(data['windfarm_yaws'])) # updates flow profile

		# Plot horizontal profile and turbine yaws
		hor_plane = fi.get_hor_plane(height=fi.floris.farm.turbines[0].hub_height, y_bounds=(-400,400), x_resolution=400, y_resolution=400)
		wfct.visualization.visualize_cut_plane(hor_plane, ax=trb_plan)
		wfct.visualization.plot_turbines(
		    ax=trb_plan,
		    layout_x=fi.layout_x,
		    layout_y=fi.layout_y,
		    yaw_angles=list(data['windfarm_yaws']),
		    D=diameter
		)

		# Add yaw labels
		for idx, yaw in enumerate(list(data['windfarm_yaws'])):
			trb_plan.text((x_spacing) * (idx) * diameter, -175, "{}$^\circ$".format(yaw), fontsize=8)

		# apply some formatting
		trb_plan.set_title("U = {} m/s, Wind Direction = {}$^\circ$" .format(data['ws'], data['wd']), fontsize=8, fontweight='bold')
		trb_plan.set_xticklabels(np.arange(-(x_spacing/2),100*x_spacing, (x_spacing/2)), fontsize=8)
		trb_plan.set_xlabel('D', fontsize=9, style='italic')
		trb_plan.set_yticklabels([])
		trb_plan.set_yticks([])
		trb_plan.tick_params(direction="out", length=2.0)
		trb_plan.set_ylim([-300,300])

		# define helper function for vertical plots
		def vert_cross(yaw_angles, diameter, x_spacing):	
		
			# calculate lagged wakes for vertical profiles
			fi.calculate_wake(yaw_angles=yaw_angles)

			cross_sections['cross_section_%s' %idx] = fi.get_cross_plane(((x_spacing) * idx * diameter) - diameter, z_bounds=(0,300), y_bounds=(-200,200))
			img = wfct.visualization.visualize_cut_plane(cross_sections['cross_section_%s' %idx], ax=wake_profiles['wake_profile_%s' %idx], minSpeed=6.0, maxSpeed=8)
			wake_profiles['wake_profile_%s' %idx].set_ylim([0, 300])

			# reverse cut so looking down-stream
			wfct.visualization.reverse_cut_plane_x_axis_in_plot(wake_profiles['wake_profile_%s' %idx])

			# apply centeral heading
			if idx == 2: #assume at lead three agents
				wake_profiles['wake_profile_%s' %idx].set_title('Vertical Cross Sections of Wake Profiles',fontsize=8, fontweight='bold')

			# plot line on plan vis to show vertical slice
			trb_plan.plot([((x_spacing) * idx * diameter) - diameter, ((x_spacing) * idx * diameter) - diameter], [-500, 500], 'k', linestyle='dashed', lw=0.6)

			# get current vertical slice distance in diameters
			current_dia = (((x_spacing) * idx * diameter) - diameter) / diameter

			# apply some formatting to the graphs
			wake_profiles['wake_profile_%s' %idx].set_xticklabels([])
			wake_profiles['wake_profile_%s' %idx].set_xlabel('{}D'.format(current_dia), fontsize=9, style='italic')
			wake_profiles['wake_profile_%s' %idx].set_yticklabels([])
			wake_profiles['wake_profile_%s' %idx].set_yticks([])

			# plot line on plan vis to show vertical slice
			wake_profiles['wake_profile_%s' %idx].plot([0, 0], [-500, 500], 'k', linestyle='dashed', lw=0.4, alpha=0.4)

			# reinstate current yaws
			fi.calculate_wake(yaw_angles=list(data['windfarm_yaws']))	


		# update vertical cross sections with time dealys 
		for idx in range(1, agents+1):
			if idx <= 2 and data['locked_turb'][idx] == False:
				vert_cross(list(data['prev_windfarm_yaws']), diameter, x_spacing)
			elif data['out_wake_delays'] <= -1.0 and idx == 3:
				vert_cross(list(data['prev_windfarm_yaws']), diameter, x_spacing)
			elif idx <= 2 and data['locked_turb'][idx] == True:
				vert_cross(list(data['intial_windfarm_yaws']), diameter, x_spacing)
			elif data['out_wake_delays'] > -1.0 and idx == 3:
				vert_cross(list(data['intial_windfarm_yaws']), diameter, x_spacing)

		# first line graph - Power + wind speed 
		wf_pwr_line.set_data(time_all, windfarm_power_all)
		wf_pwr_line_sec.set_data(time_all, ws_all)

		# second line graph - Turbine Yaw Angles + reward
		for line, key in enumerate(turbine_yaws_all.keys()):
			turb_yaw_lines[line].set_data(time_all, turbine_yaws_all[key])

		trb_yaws_sec.set_data(time_all, reward_all)


		# third line graph turbine powers - left axis first turbine, then others on right
		trb_pwrs_line.set_data(time_all, first_turbine_powers_all) # left axis first turbine 

		for line, key in enumerate(turbine_powers_all.keys()): # right axis other turbines
			sec_trb_pwrs_lines[line].set_data(time_all, turbine_powers_all[key])


		return [wf_pwr_line, trb_yaw_line, trb_pwrs_line]

	
	animation_ = FuncAnimation(fig, update, interval=24, blit=True, save_count=10)
	plt.show(block=True)	
	animation_.save('floris_animation.gif', writer='imagemagick', fps=20)

	# FFwriter = animation.FFMpegWriter(fps=20, codec="libx264")     
	# animation_.save('test.mp4', writer = FFwriter)
	# fig.savefig('floris_final_still.jpg', bbox_inches='tight')







































