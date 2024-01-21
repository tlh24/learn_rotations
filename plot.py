import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import time
import pdb

def load_data_from_db(db_name, table_name):
	conn = sqlite3.connect(db_name)
	cursor = conn.cursor()

	# Select M and SNR from the table
	cursor.execute(f'SELECT VARIATE, ALGO, SNR FROM "{table_name}"')
	data = cursor.fetchall()
	
	conn.close()
	return data
	
def process_data(data): 
	# Group data into a dictionary indexed by algorithm
	# each dictionary element is a list of (m,snr) tuples
	algodata = {}
	for m, algo, snr in data:
		if algo not in algodata:
			algodata[algo] = []
		algodata[algo].append((m, snr))
		
	for n in algodata: 
		algodata[n].sort()
		
	return algodata

def plot_data(db_name, table_names):
	optim_names = ['Adadelta','Adagrad','Adam','Adamw','Adamax','ASGD','NAdam','RAdam','RMSprop','Rprop','SGD','Lion','SVD']
	optim_index = [*range(13)]
	optim_index.remove(2) # don't plot Adam -- same as AdamW with wd=0
	
	colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
	if len(table_names) > len(colors):
		raise ValueError("Not enough colors defined for the number of tables")
	
	all_data = [process_data(load_data_from_db(db_name, table)) for table in table_names]
	# list of dictionaries (descibed above)

	plot_rows = 3
	plot_cols = 4
	figsize = (20, 15)
	fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize, layout='constrained')
	ymax = np.zeros(13)
	for i,algo in enumerate(optim_index): 
		for t,d in enumerate(all_data): 
			r = i // 4
			c = i % 4
			os, snr = zip(*d[algo])
			label = ''
			if i == 0: 
				label=table_names[t]
			ax[r,c].semilogx(os, snr, color=colors[t], label=label)
			ax[r,c].set_title(f"{optim_names[algo]}")
			ax[r,c].set_xlabel("rank")
			ax[r,c].tick_params(left=True)
			ax[r,c].tick_params(right=True)
			ax[r,c].tick_params(bottom=True)
			ymax[i] = max(ymax[i], np.max(snr))
			ax[r,c].set_ylim(ymin=0, ymax=ymax[i]+5.0)

	fig.legend(loc='outside lower center')
	plt.show()

# table_names = ["variable_Mrank_unscaled_sv", "variable_Mrank_scaled_sv", "fixed_M_variable_rank_unscaled_sv", "fixed_M_variable_rank_unscaled_sv_2k", "fixed_M_variable_N_unscaled_sv", "variable_M_fixed_N_unscaled_sv", "fixed_M_variable_N_variable_R_scaled_sv", "variable_M_fixed_N_fixed_R_scaled_sv", "variable_MNR_scaled_sv_2k"]
# table_names = ["fixed_M_variable_N_unscaled_sv"]
# "variable_Mrank_ortho2", "variable_Mrank_ortho", "variable_Mrank_unscaled_sv2", "variable_Mrank_scaled_sv2"
table_names = ['variable_MNR_unscaled_sv_500', 'variable_MNR_scaled_sv_500', 'variable_M_fixed_NR_unscaled_sv_500', 'fixed_M_variable_NR_unscaled_sv_500', 'fixed_M_fixed_N_variable_R_unscaled_sv_500', 'variable_M_fixed_NR_scaled_sv_500', 'fixed_M_variable_NR_scaled_sv_500', 'fixed_M_fixed_N_variable_R_scaled_sv_500']
plot_data("snr2.db", table_names)
