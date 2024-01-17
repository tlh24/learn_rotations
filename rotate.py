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

batch_size = 32

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

def make_ortho_matrix(M, N):
	dim = max(M,N)
	phi = torch.tensor(ortho_group.rvs(dim=dim))
	return phi[:M, :N]

# def make_rr_matrix(siz, rank):
# 	# make a non-orthogonal normal matrix with rank less than size
# 	o = torch.tensor(ortho_group.rvs(dim=siz), dtype=torch.float32)
# 	if siz > rank:
# 		o = o[:, :rank]
# 		r = torch.randn(rank, siz-rank)
# 		m = o @ r
# 		m = torch.nn.functional.normalize(m, dim=0, p=2)
# 		o = torch.cat((o,m), dim=1)
# 		# # test -- result is not normed??
# 		# x = random_unit_vectors(32, siz, o.device)
# 		# x = x.unsqueeze(1)
# 		# y = x @ o
# 		# y = y.squeeze()
# 		# print(torch.norm(y, dim=1))
# 	return o

# def make_rrz_matrix(siz, rank):
# 	# make a non-orthogonal normal matrix with rank less than size
# 	# -- extra columns are zeroed
# 	o = torch.tensor(ortho_group.rvs(dim=siz), dtype=torch.float32)
# 	m = torch.zeros(siz,siz)
# 	m[:,:rank] = o[:,:rank]
# 	return m

# Rotation matrix phi is M by N 
# Input x is dim M 
# Output y is dim N
# -- when M = N, this is a square orthonormal matrix, full rank.
# -- when M < N, drop rows from phi -> y is in rank M subspace of R^N
# -- when M > N, drop columns from phi -> 
#		x and y are both full rank (M and N respectively). 
# eplen is the length of the episode

def train(phi, M, N, eplen, algo, scl, device):
	repeats = phi.shape[0]
	trainlosses = np.zeros((repeats,eplen))
	testlosses = np.zeros((repeats,eplen))
	
	for k in range(repeats): 
		m = phi[k,:,:]
		
		model = Net(M,N).to(device)
		lr = 0.01
		wd = 0.01
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
				optimizer = optim.SGD(model.parameters(), lr=lr*10, weight_decay=wd)
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
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # essential!
			# otherwise, get oscillations, seemingly due to term-coupling.
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
	if algo == 0 :
		x = random_unit_vectors(M*2, M, device=torch.device("cpu"))
		# x = torch.nn.functional.pad(x, (0,fxsiz-siz), "constant", 1)
		# x = random_unit_vectors(siz*3, siz, device=torch.device("cpu"))
		m_ = m.cpu().numpy() # re-use the matrix
		y = x @ m_
		u, s, vh = np.linalg.svd(y)
		rank = np.sum(s > 1e-4)
		print(f'rank of mapping: {rank}; [{M},{N}]')
		mp = np.linalg.lstsq(x, y, rcond=None)[0]
		x = random_unit_vectors(2*M, M, device=torch.device("cpu"))
		y = x @ m_
		p = x @ mp
		loss = torch.sum((y - p)**2)
		print(f"loss on via linear regression {loss} size [{M},{N}]")
	else:
		if algo == 11:
			print(f'done with [{M},{N}]')
	return trainlosses, testlosses, name

def linalg_test(): 
	N = 32
	for M in range(4,32): 
		phi = torch.zeros((M, N), device=torch.device("cpu"), dtype=torch.float32)
		phi = make_ortho_matrix(M, N).to(dtype=torch.float)
		x = random_unit_vectors(16, M, device=torch.device("cpu"))
		y = x @ phi
		u, s, vh = np.linalg.svd(y)
		rank = np.sum(s > 1e-4)
		print(f'rank of mapping: {rank}; M:{M},N:{N}')
		mp = np.linalg.lstsq(x, y, rcond=None)[0]
		x = random_unit_vectors(2*M, M, device=torch.device("cpu"))
		y = x @ phi
		p = x @ mp
		loss = torch.sum((y - p)**2)
		print(f"loss on via linear regression {loss} size [{M},{N}]")
		
def calc_scale(M, N, device): 
	scl = np.zeros((16,))
	for i in range(16): 
		phi = make_ortho_matrix(M, N).to(device=device, dtype=torch.float32)
		x = random_unit_vectors(batch_size, M, device)
		x = x.unsqueeze(1)
		y = x @ phi
		l = torch.sqrt(torch.sum(y**2, -1)) # l2 norm
		scl[i] = torch.mean(l).cpu().item()
	return 1.0 / np.mean(scl)

def do_plot(M, N, repeats, device):
	plot_rows = 3
	plot_cols = 4
	figsize = (18, 15)
	fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	
	# normalize the l2 norm of y (x is always a unit vector)
	if M != N:
		scl = calc_scale(M, N, device)
	else:
		# unit normal vectors* orthonormal matrix -> normal vectors.
		scl = 1.0
	
	phi = torch.zeros((repeats, M, N), device=device, dtype=torch.float32)
	# use the same phi matrices for all algorithms. 
	for i in range(repeats):
		phi[i,:,:] = make_ortho_matrix(M, N)
	
	for algo in range(12):
		trainlosses,testlosses,name = train(phi, M, N, 500, algo, scl, device)
		r = algo // 4
		c = algo % 4
		ax[r,c].plot(np.log(trainlosses.T))
		ax[r,c].plot(np.log(testlosses.T))
		ax[r,c].set_title(f"{name} [{M},{N}]")
		if M > N: 
			ax[r,c].set_ylim(ymin=-7, ymax=1.5)
		if M < N: 
			ax[r,c].set_ylim(ymin=-20, ymax=5.0)
		if M == N: 
			ax[r,c].set_ylim(ymin=-20, ymax=1.5)
		ax[r,c].tick_params(left=True)
		ax[r,c].tick_params(right=True)
		ax[r,c].tick_params(bottom=True)

	if M > N:
		fig.savefig(f'fixed_M_variable_N/rotate_loss__wd1e-2_size{N}.png', bbox_inches='tight')
	if M < N: 
		fig.savefig(f'variable_M_fixed_N/rotate_loss__wd1e-2_size{M}.png', bbox_inches='tight')
	if M == N: 
		fig.savefig(f'variable_M_variable_N/rotate_loss__wd1e-2_size{M}.png', bbox_inches='tight')
	plt.close(fig)

if __name__ == '__main__':
	# # Initialize parser
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--offset", type=int, choices=range(0,10), default=0, help="set the offset for iteration")
	parser.add_argument("-l", "--lod", type=int, choices=range(0,10), default=2, help="set the level of detail")
	parser.add_argument("-c", "--cuda", type=int, choices=range(0,2), default=0, help="set the CUDA device")
	args = parser.parse_args()
	o = args.offset
	lod = args.lod
	repeats = 5

	device = torch.device(type='cuda', index=args.cuda)
	
	# for siz in range(1019, 1024):
	# 	do_plot(N, siz, repeats, device)
	def run_variableLOD(fun): 
		for P in range(4+o,64,lod):
			fun(P)
		for P in range(64+o*2,128,lod*2):
			fun(P)
		for P in range(128+o*4,256,lod*4):
			fun(P)
		for P in range(256+o*8,512,lod*8):
			fun(P)
		for P in range(512+o*16,1024,lod*16):
			fun(P)
			
	def innerfun(N):
		M = 1024
		do_plot(M, N, repeats, device)
	
	run_variableLOD(innerfun)
