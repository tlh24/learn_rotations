import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import matplotlib.pyplot as plt
import numpy as np
import math
from lion_pytorch import Lion
from scipy.stats import ortho_group
import argparse
import sqlite3
import time
import os

batch_size = 32

def create_database_and_table(db_name, table_name):
	# Connect to the SQLite database
	conn = sqlite3.connect(db_name)
	cursor = conn.cursor()

	# Create table
	cursor.execute(f'''CREATE TABLE IF NOT EXISTS "{table_name}" 
							(VARIATE INTEGER, ALGO INTEGER, SNR REAL, UNIQUE(VARIATE, ALGO))''')
	conn.commit()
	conn.close()

def insert_data(db_name, table_name, variate, algo, snr):
	try:
		# Connect to the SQLite database
		conn = sqlite3.connect(db_name)
		cursor = conn.cursor()

		# Insert a row of data
		cursor.execute(f'REPLACE INTO "{table_name}" (VARIATE, ALGO, SNR) VALUES (?, ?, ?)', (variate, algo, snr))
		conn.commit()
	except sqlite3.OperationalError as e:
		# Handle database is locked error
		if "database is locked" in str(e):
			time.sleep(0.25)  # Wait before retrying
			insert_data(db_name, table_name, variate, algo, snr)
		else: 
			print(e)
			exit()
	finally:
		conn.close()

class Net(nn.Module): 
	def __init__(self, M,N):
		super(Net, self).__init__()
		self.l1 = nn.Linear(M, N, bias = True)
		# bias is not required & does not affect gradient oscillations
		# torch.nn.init.zeros_(self.l1.weight)
		# torch.nn.init.zeros_(self.l1.bias)
		
	def forward(self, x): 
		y = self.l1(x)
		return y
		

class NetTrig(nn.Module): 
	# yes, parameterizing a 2x2 rotation matrix with one parameter 
	# does work -- backprop can learn theta!
	def __init__(self):
		super(NetTrig, self).__init__()
		self.theta = nn.Parameter(torch.zeros(1))
		
	def forward(self, x): 
		t = self.theta
		m = torch.zeros(2,2)
		m[0,0] = torch.cos(t)
		m[1,1] = torch.cos(t)
		m[0,1] = -torch.sin(t)
		m[1,0] = torch.sin(t)
		return x @ m

def random_unit_vectors(batch_size, siz, device):
	x = torch.rand(batch_size, siz, device=device) - 0.5
	x = torch.nn.functional.normalize(x, dim=1)
	return x

def make_rot_matrix(siz):
	o = torch.eye(siz)
	s = siz
	if siz == 2:
		s = 1
	for i in range(s):
		m = torch.zeros(siz, siz)
		phi = np.random.rand(1).item() * np.pi * 2.0
		j = (i+1) % siz
		m[i,i] = math.cos(phi)
		m[i,j] = -math.sin(phi)
		m[j,i] = math.cos(phi)
		m[j,j] = math.sin(phi)
		o = o @ m
	return o

# def make_ortho_matrix(M, N):
# 	dim = max(M,N)
# 	phi = torch.tensor(ortho_group.rvs(dim=dim))
# 	return phi[:M, :N]

def make_rr_matrix(M, N, R, scale_sv):
	# make a non-orthogonal normal matrix with rank less than size
	dim = max(M,N)
	o = torch.tensor(ortho_group.rvs(dim=dim), dtype=torch.float32)
	if R < dim // 2: 
		p = torch.flip(o, dims=(1,)) # flip columns/re-use.
	else:
		p = torch.tensor(ortho_group.rvs(dim=dim), dtype=torch.float32)
	if scale_sv: # scale the singular values.
		s = torch.diag(torch.arange(R) / R * 4 + 1)
	else: 
		s = torch.eye(R)
	phi = o[:,:R] @ s @ p[:, :R].T
	return phi[:M, :N]

# Rotation matrix phi is M by N 
# Input x is dim M 
# Output y is dim N
# -- when M = N, this is a square orthonormal matrix, full rank.
# -- when M < N, drop rows from phi -> y is in rank M subspace of R^N
# -- when M > N, drop columns from phi -> 
#		x and y are both full rank (M and N respectively). 
# eplen is the length of the episode

def train(phi, M, N, R, eplen, algo, scl, device):
	repeats = phi.shape[0]
	trainlosses = np.zeros((repeats,eplen))
	testlosses = np.zeros((repeats,eplen))
	
	for k in range(repeats): 
		m = phi[k,:,:]
		
		model = Net(M,N).to(device)
		lr = 0.01
		wd = 0.00
		if algo < 12:
			match algo:
				case 0:
					optimizer = optim.Adadelta(model.parameters(), lr=lr*50, weight_decay=wd)
					name = "Adadelta"
				case 1:
					optimizer = optim.Adagrad(model.parameters(), lr=lr*10, weight_decay=wd, lr_decay=0.001)
					name = "Adagrad"
				case 2:
					optimizer = optim.Adam(model.parameters(), lr=lr)
					name = "Adam"
				case 3:
					optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
					name = "AdamW"
				case 4:
					optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
					name = "Adamax"
				case 5:
					optimizer = optim.ASGD(model.parameters(), lr=lr, weight_decay=wd)
					name = "ASGD"
				case 6:
					optimizer = optim.NAdam(model.parameters(), lr=lr, weight_decay=wd)
					name = "NAdam"
				case 7:
					optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=wd)
					name = "RAdam"
				case 8:
					optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, alpha=0.90)
					name = "RMSprop"
				case 9:
					optimizer = optim.Rprop(model.parameters(), lr=lr)
					name = "Rprop"
				case 10:
					optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
					name = "SGD"
				case 11:
					optimizer = Lion(model.parameters(), lr=lr, weight_decay=wd)
					name = "Lion"
			
			for i in range(eplen): 
				x = random_unit_vectors(batch_size, M, device)
				x = x.unsqueeze(1)
				y = (x @ m) * scl
				model.zero_grad()
				p = model(x)
				loss = torch.sum((y - p)**2)
				loss.backward()
				optimizer.step()
				trainlosses[k,i] = loss.detach() / batch_size
				
				with torch.no_grad(): 
					# new data!
					x = random_unit_vectors(batch_size, M, device)
					x = x.unsqueeze(1)
					y = (x @ m) * scl
					p = model(x)
					loss = torch.sum((y - p)**2)
					testlosses[k,i] = loss.detach() / batch_size
			
		# do the same for linear regression (positive control)
		if algo == 12 and k == 0:
			name = "SVD"
			x = random_unit_vectors(M*2, M, device=torch.device("cpu"))
			# x = torch.nn.functional.pad(x, (0,fxsiz-siz), "constant", 1)
			# x = random_unit_vectors(siz*3, siz, device=torch.device("cpu"))
			m_ = m.cpu().numpy() # re-use the matrix
			y = x @ m_
			u, s, vh = np.linalg.svd(x, full_matrices=False)
			# rank = np.sum(s > 1e-4)
			# print(f'rank of mapping: {rank}; [{M},{N}]{R}') # check! 
			s = s + (s < 1e-4) * 1e32
			sp = 1 / s
			mp = vh.T @ np.diag(sp) @ u.T @ y.numpy()
			# mq = np.linalg.lstsq(x, y, rcond=None)[0] # same  results
			x = random_unit_vectors(2*M, M, device=torch.device("cpu"))
			y = x @ m_
			p = x @ mp
			# q = x @ mp
			nulloss = torch.sum((y)**2)
			testlosses[k,0] = nulloss / batch_size # for snr calc
			loss = torch.sum((y - p)**2)
			# lossq = torch.sum((y - q)**2)
			testlosses[k,1] = loss / batch_size
			print(f"loss on via linear regression {loss}; done with [{M},{N}] r:{R}")
	
	return trainlosses,testlosses,name

# def linalg_test(): 
# 	N = 32
# 	for M in range(4,32): 
# 		phi = make_ortho_matrix(M, N).to(dtype=torch.float)
# 		x = random_unit_vectors(16, M, device=torch.device("cpu"))
# 		y = x @ phi
# 		u, s, vh = np.linalg.svd(y)
# 		rank = np.sum(s > 1e-4)
# 		print(f'rank of mapping: {rank}; M:{M},N:{N}')
# 		mp = np.linalg.lstsq(x, y, rcond=None)[0]
# 		x = random_unit_vectors(2*M, M, device=torch.device("cpu"))
# 		y = x @ phi
# 		p = x @ mp
# 		loss = torch.sum((y - p)**2)
# 		print(f"loss on via linear regression {loss} size [{M},{N}]")
		
def calc_scale(M, N, R, scale_sv, device): 
	scl = np.zeros((16,))
	for i in range(16): 
		phi = make_rr_matrix(M, N, R, scale_sv).to(device=device, dtype=torch.float32)
		x = random_unit_vectors(batch_size, M, device)
		x = x.unsqueeze(1)
		y = x @ phi
		l = torch.sqrt(torch.sum(y**2, -1)) # l2 norm
		scl[i] = torch.mean(l).cpu().item()
	ms = 1.0 / np.mean(scl)
	print(f"scale {ms}")
	return ms

def do_plot(M, N, R, scale_sv, variate, repeats, db_name, fdir, eplen, device):
	plot_rows = 3
	plot_cols = 4
	figsize = (18, 15)
	fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	
	lossall = np.zeros((13,repeats,eplen))
	
	# normalize the l2 norm of y (x is always a unit vector)
	scl = 1.0
	if M != N or M != R or N != R: 
		scl = calc_scale(M, N, R, scale_sv, device)
	
	phi = torch.zeros((repeats, M, N), device=device, dtype=torch.float32)
	# use the same phi matrices for all algorithms. 
	for i in range(repeats):
		phi[i,:,:] = make_rr_matrix(M, N, R, scale_sv)
	
	names = []
	for algo in range(13):
		trainlosses,testlosses,name = train(phi, M, N, R, eplen, algo, scl, device)
		lossall[algo,:,:] = testlosses
		names.append(name)
		if algo < 12: 
			r = algo // 4
			c = algo % 4
			ax[r,c].plot(np.log(trainlosses.T))
			ax[r,c].plot(np.log(testlosses.T))
			ax[r,c].set_title(f"{name} [{M},{N}] rank {R}")
			ax[r,c].set_ylim(ymin=-20, ymax=5.0)
			ax[r,c].tick_params(left=True)
			ax[r,c].tick_params(right=True)
			ax[r,c].tick_params(bottom=True)

	fig.savefig(f'{fdir}/loss_rank{variate}.png', bbox_inches='tight')
	plt.close(fig)
	snr = np.zeros((13,))
	sta = np.mean(lossall[:,:,0], axis=(1,))
	fin = np.mean(lossall[:,:,-4:], axis=(1,2))
	sta[12] = np.mean(lossall[12,0,0]) # only one repeat
	fin[12] = np.mean(lossall[12,0,1])
	sta = np.mean(sta) # average all the starting losses to reduce noise
	snr = 10 * (np.log10(sta) - np.log10(fin))
	for algo in range(13): 
		insert_data(db_name, fdir, variate, algo, snr[algo])
		
	return lossall,names


if __name__ == '__main__':
	# # Initialize parser
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--offset", type=int, choices=range(0,20), default=0, help="set the offset for iteration")
	parser.add_argument("-l", "--lod", type=int, choices=range(0,20), default=2, help="set the level of detail")
	parser.add_argument("-r", "--repeats", type=int, choices=range(1,10), default=4, help="number of replicates")
	parser.add_argument("-c", "--cuda", type=int, choices=range(0,2), default=0, help="set the CUDA device")
	parser.add_argument("-e", "--episodelength", type=int, choices=range(2,30), default=20, help="set the training length, units of 100 so 5 -> 500 steps")
	parser.add_argument("-m", "--mode", type=int, choices=range(0,4), default=0, help="set the test mode. 0-3, see source.")
	parser.add_argument("-s", "--scalesv", type=int, choices=range(2), default=0, help="turn off / on singular value scaling")
	args = parser.parse_args()
	o = args.offset
	lod = args.lod
	repeats = args.repeats
	eplen = args.episodelength * 100
	scale_sv = args.scalesv

	device = torch.device(type='cuda', index=args.cuda)
	db_name = "snr4.db"
	
	print(f"RUN: offset:{o} lod:{lod} repeats:{repeats} eplen:{eplen} cuda:{args.cuda} mode:{args.mode} scalesv:{scale_sv}")
	
	def run_variableLOD(fun): 
		lossall = []
		if o < 2: 
			fun(2+o)
		for P in range(4+o,64,lod):
			fun(P)
		for P in range(64+o*2,128,lod*2):
			fun(P)
		for P in range(128+o*4,256,lod*4):
			fun(P)
		for P in range(256+o*8,1025,lod*8):
			fun(P)
			
	def run_test(fdir, fun): 
		os.makedirs(fdir, exist_ok=True)
		create_database_and_table(db_name, fdir)
		run_variableLOD(fun)
	
	scaling = "unscaled_sv"
	if scale_sv:
		scaling = "scaled_sv"
	match args.mode: 
		case 0: 
			fdir = f"variable_MNR_{scaling}_{eplen}"
			def inner(variate):
				M = variate
				N = variate
				R = variate
				do_plot(M, N, R, scale_sv, variate, repeats, db_name, fdir, eplen, device)
			run_test(fdir,inner)
		case 1: 
			fdir = f"variable_M_fixed_NR_{scaling}_{eplen}"
			def inner(variate):
				M = variate
				N = 1024
				R = 1024
				do_plot(M, N, R, scale_sv, variate, repeats, db_name, fdir, eplen, device)
			run_test(fdir,inner)
		case 2: 
			fdir = f"fixed_M_variable_NR_{scaling}_{eplen}"
			def inner(variate):
				M = 1024
				N = variate
				R = variate
				do_plot(M, N, R, scale_sv, variate, repeats, db_name, fdir, eplen, device)
			run_test(fdir,inner)
		case 3: 
			fdir = f"fixed_M_fixed_N_variable_R_{scaling}_{eplen}"
			def inner(variate):
				M = 1024
				N = 1024
				R = variate
				do_plot(M, N, R, scale_sv, variate, repeats, db_name, fdir, eplen, device)
			run_test(fdir,inner)
		
