import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import time
import sys
import math
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

def plot_data(db_name, table_names, eplen):
	optim_names = ['Adadelta','Adagrad','Adam','AdamW','Adamax','ASGD','NAdam','RAdam','RMSprop','Rprop','SGD','Lion','SVD']
	optim_index = [*range(13)]
	optim_index.remove(2) # don't plot Adam -- same as AdamW with wd=0
	
	colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
	linestyles = ['solid','dashed']
	if len(table_names)//2 > len(colors):
		raise ValueError("Not enough colors defined for the number of tables")
	
	all_data = [process_data(load_data_from_db(db_name, table)) for (table,_) in table_names]
	# list of dictionaries (descibed above)

	plt.rcParams.update({'font.size': 18})
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
				_,label=table_names[t]
			ax[r,c].semilogx(os, snr, color=colors[t//2], label=label, linestyle=linestyles[t%2])
			ax[r,c].set_title(f"{optim_names[algo]}")
			# ax[r,c].set_xlabel("V") distracting, put in figure legend.
			ax[r,c].tick_params(left=True)
			ax[r,c].tick_params(right=True)
			ax[r,c].tick_params(bottom=True)
			ymax[i] = max(ymax[i], np.max(snr))
			ax[r,c].set_ylim(ymin=0, ymax=ymax[i]+5.0)

			# add in a 'chinchilla' line @ 20x # params.
			if t % 2 == 0:
				match t//2:
					case 0:
						chinc = math.sqrt(eplen * 32 / 20.0) # 56.5
					case 1:
						chinc = (eplen * 32 / 20.0) / 1024 # e.g. 3.125
					case 2:
						chinc = (eplen * 32 / 20.0) / 1024
					case 3:
						chinc = 1024 # need 20M samples; clip right
				if chinc < 1024 and chinc > 2.0 :
					ax[r,c].semilogx([chinc,chinc], [0.0, ymax[i]], color=colors[t//2], linestyle='dotted')
	fig.legend(loc='outside lower center')
	fig.savefig(f'{db_name}.eps', format='eps', bbox_inches='tight')
	plt.show()

# table_names = ["variable_Mrank_unscaled_sv", "variable_Mrank_scaled_sv", "fixed_M_variable_rank_unscaled_sv", "fixed_M_variable_rank_unscaled_sv_2k", "fixed_M_variable_N_unscaled_sv", "variable_M_fixed_N_unscaled_sv", "fixed_M_variable_N_variable_R_scaled_sv", "variable_M_fixed_N_fixed_R_scaled_sv", "variable_MNR_scaled_sv_2k"]
# table_names = ["fixed_M_variable_N_unscaled_sv"]
# "variable_Mrank_ortho2", "variable_Mrank_ortho", "variable_Mrank_unscaled_sv2", "variable_Mrank_scaled_sv2"
eplen = int(sys.argv[1])
eplen_s = f"sv_{eplen}"
db_name = f"snr_{eplen}.db"
table_names = [ \
	(f'variable_MNR_unscaled_{eplen_s}','[V,V] rank:V unscaled'),
	(f'variable_MNR_scaled_{eplen_s}','[V,V] rank:V scaled'), (f'variable_M_fixed_NR_unscaled_{eplen_s}','[V,1024] rank:V unscaled'),
	(f'variable_M_fixed_NR_scaled_{eplen_s}','[V,1024] rank:V scaled'), (f'fixed_M_variable_NR_unscaled_{eplen_s}','[1024,V] rank:V unscaled'),
	(f'fixed_M_variable_NR_scaled_{eplen_s}','[1024,V] rank:V scaled'), (f'fixed_M_fixed_N_variable_R_unscaled_{eplen_s}','[1024,1024] rank:V unscaled'),
	(f'fixed_M_fixed_N_variable_R_scaled_{eplen_s}','[1024,1024] rank:V scaled'),]
plot_data(db_name, table_names, eplen)
