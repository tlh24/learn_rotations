import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import time
import pdb

def load_data_from_db(db_name, table_name):
	conn = sqlite3.connect(db_name)
	cursor = conn.cursor()

	# Select M and SNR from the table
	cursor.execute(f'SELECT M, ALGO, SNR FROM "{table_name}"')
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
	
	
	colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
	if len(table_names) > len(colors):
		raise ValueError("Not enough colors defined for the number of tables")
	
	all_data = [process_data(load_data_from_db(db_name, table)) for table in table_names]
	# list of dictionaries (descibed above)

	plot_rows = 3
	plot_cols = 5
	figsize = (20, 15)
	fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	for algo in range(13): 
		for t,d in enumerate(all_data): 
			r = algo // 5
			c = algo % 5
			os, snr = zip(*d[algo])
			ax[r,c].semilogx(os, snr, color=colors[t])
			ax[r,c].set_title(f"{optim_names[algo]}")
			ax[r,c].set_xlabel("rank")
			ax[r,c].tick_params(left=True)
			ax[r,c].tick_params(right=True)
			ax[r,c].tick_params(bottom=True)
			ax[r,c].set_ylim(ymin=0)

	plt.show()

table_names = ["variable_Mrank_unscaled_sv", "variable_Mrank_ortho",  "fixed_M_variable_rank_unscaled_sv", "fixed_M_variable_rank_unscaled_sv_2k"]
# "variable_Mrank_ortho2",
plot_data("snr.db", table_names)
