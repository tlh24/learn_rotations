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

batch_size = 32

class Net(nn.Module): 
	def __init__(self, siz):
		super(Net, self).__init__()
		self.l1 = nn.Linear(siz, siz, bias = True)
		# bias is not required & does not affect gradient oscillations
		# torch.nn.init.zeros_(self.l1.weight)
		# torch.nn.init.zeros_(self.l1.bias)
		
	def forward(self, x): 
		x = self.l1(x)
		return x
		

class NetTrig(nn.Module): 
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
	if siz == 2 and False:
		theta = torch.rand(batch_size) * math.pi * 2.0
		x = torch.stack((torch.sin(theta), torch.cos(theta)), 1)
	else:
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

def make_ortho_matrix(siz):
	return torch.tensor(ortho_group.rvs(dim=siz))

def make_rr_matrix(siz, rank):
	# make a non-orthogonal normal matrix with rank less than size
	o = torch.tensor(ortho_group.rvs(dim=siz), dtype=torch.float32)
	if siz > rank:
		o = o[:, :rank]
		r = torch.randn(rank, siz-rank)
		m = o @ r
		m = torch.nn.functional.normalize(m, dim=0, p=2)
		o = torch.cat((o,m), dim=1)
		# # test -- result is not normed??
		# x = random_unit_vectors(32, siz, o.device)
		# x = x.unsqueeze(1)
		# y = x @ o
		# y = y.squeeze()
		# print(torch.norm(y, dim=1))
	return o

def make_rrz_matrix(siz, rank):
	# make a non-orthogonal normal matrix with rank less than size
	# -- extra columns are zeroed
	o = torch.tensor(ortho_group.rvs(dim=siz), dtype=torch.float32)
	m = torch.zeros(siz,siz)
	m[:,:rank] = o[:,:rank]
	return m

def train(phi, N, algo, siz, device):
	repeats = phi.shape[0]
	losses = np.zeros((repeats,N))
	gradients = np.zeros((repeats,N,4))
	weights = np.zeros((repeats,N,4))
	fxsiz = phi.shape[1]
	
	for k in range(repeats): 
		m = phi[k,:,:]
		
		model = Net(fxsiz).to(device)
		# model = NetTrig()
		lr = 0.01
		wd = 0.01
		match algo:
			case 0:
				optimizer = optim.Adadelta(model.parameters(), lr=lr*20, weight_decay=wd)
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
		
		for i in range(N): 
			x = random_unit_vectors(batch_size, siz, device)
			x = torch.nn.functional.pad(x, (0,fxsiz-siz), "constant", 0)
				# if phi is full rank, fxsiz-siz = 0 and no padding happens.
				# if siz < fxsiz, pad it so @ works. 
				# this is a dumb way of reducing the rank of the output - but it works. 
			x = x.unsqueeze(1)
			scl = 1
			# scl = math.sqrt(siz*1.0)
			y = (x @ m) * scl
			model.zero_grad()
			p = model(x)
			loss = torch.sum((y - p)**2)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # essential!
			# otherwise, get oscillations, seemingly due to term-coupling.
			optimizer.step()
			losses[k,i] = loss.detach() / (batch_size * scl)
			# vectors are length 1, so don't divide by siz
			# gradients[k,i,:] = model.l1.weight.grad[0:2,0:2].detach().cpu().flatten()
			# weights[k,i,:] = model.l1.weight[0:2,0:2].detach().cpu().flatten()
			
	# do the same for linear regression (positive control)
	if algo == 0 and repeats == 1:
		x = random_unit_vectors(batch_size, siz, device=torch.device("cpu"))
		x = torch.nn.functional.pad(x, (0,fxsiz-siz), "constant", 1)
		# x = random_unit_vectors(siz*3, siz, device=torch.device("cpu"))
		m_ = m.cpu().numpy() # re-use the matrix
		y = x @ m_
		u, s, vh = np.linalg.svd(y)
		rank = np.sum(s > 1e-4)
		print(f'rank of y: {rank}/{siz}')
		mp = np.linalg.lstsq(x, y, rcond=None)[0]
		p = x @ mp
		loss = torch.sum((y - p)**2)
		print(f"loss on 3 batches via linear regression {loss} size {siz}")
	else:
		if algo == 0:
			print(f'done with {siz}')
	return losses, gradients, weights, name

def do_plot(N, siz, repeats, device):
	plot_rows = 3
	plot_cols = 4
	figsize = (18, 15)
	fxsiz = 1024
	fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	# fig2, ax2 = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	# fig3, ax3 = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	phi = torch.zeros((repeats, fxsiz, fxsiz), device=device, dtype=torch.float32)
	for i in range(repeats):
		phi[i,:,:] = make_ortho_matrix(fxsiz) # siz | fixsiz
		# phi[i,:,:,] = make_rrz_matrix(siz, 4)
	for algo in range(12):
	# for algo in [0,1,2,7,8]:
		losses,grads,weights,name = train(phi, N, algo, siz, device)
		r = algo // 4
		c = algo % 4
		ax[r,c].plot(np.log(losses.T))
		ax[r,c].set_title(name)
		# ax[r,c].set_ylim(ymin=-10, ymax=2) # unscaled
		# ax[r,c].set_ylim(ymin=-8, ymax=5) # sqrt scaled
		ax[r,c].tick_params(left=True)
		ax[r,c].tick_params(right=True)
		ax[r,c].tick_params(bottom=True)

		# ax2[r,c].plot(grads[0,:,:])
		# ax2[r,c].set_title(name)
		# ax3[r,c].plot(weights[0,:,:])
		# ax3[r,c].set_title(name)

	# plt.show()
	fig.savefig(f'fixed_size_variable_rank/rotate_loss__wd1e-2_size{siz}.png', bbox_inches='tight')
	# fig2.savefig(f'rotate_grads_size{siz}.png', bbox_inches='tight')
	# fig3.savefig(f'rotate_weights_size{siz}.png', bbox_inches='tight')
	plt.close(fig)
	# plt.close(fig2)
	# plt.close(fig3)

import argparse

if __name__ == '__main__':
	# # Initialize parser
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--offset", type=int, choices=range(0,10), help="set the offset for iteration")
	parser.add_argument("-l", "--lod", type=int, choices=range(0,10), help="set the level of detail")
	parser.add_argument("-c", "--cuda", type=int, choices=range(0,2), help="set the CUDA device")
	args = parser.parse_args()
	o = args.offset
	lod = args.lod
	repeats = 5
	N = 300 # duration of training

	device = torch.device(type='cuda', index=args.cuda)
	# for siz in range(1019, 1024):
	# 	do_plot(N, siz, repeats, device)
	for siz in range(4+o,64,lod):
		do_plot(N, siz, repeats, device)
	for siz in range(64+o*2,128,lod*2):
		do_plot(N, siz, repeats, device)
	for siz in range(128+o*4,256,lod*4):
		do_plot(N, siz, repeats, device)
	for siz in range(256+o*8,512,lod*8):
		do_plot(N, siz, repeats, device)
	for siz in range(512+o*16,1024,lod*16):
		do_plot(N, siz, repeats, device)
