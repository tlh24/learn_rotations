import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import matplotlib.pyplot as plt
import numpy as np
import math
from lion_pytorch import Lion

class Net(nn.Module): 
	def __init__(self):
		super(Net, self).__init__()
		self.l1 = nn.Linear(2, 2, bias = True)
		# torch.nn.init.zeros_(self.l1.weight)
		# torch.nn.init.zeros_(self.l1.bias)
		
	def forward(self, x): 
		x = self.l1(x)
		return x
		

def main(phi, N, algo): 
	batch_size = 16
	repeats = phi.shape[0]
	losses = np.zeros((repeats,N))
	
	for k in range(repeats): 
		m = torch.tensor([[math.cos(phi[k]), -math.sin(phi[k])],[math.sin(phi[k]), math.cos(phi[k])]])
		
		model = Net()
		lr = 0.01
		wd = 0.01
		match algo:
			case 0:
				optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=wd)
				name = "Adadelta"
			case 1:
				optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd, lr_decay=0.001)
				name = "Adagrad"
			case 2:
				optimizer = optim.Adam(model.parameters(), lr=lr/20)
				name = "Adam"
			case 3:
				optimizer = optim.AdamW(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "AdamW"
			case 4:
				optimizer = optim.Adamax(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "Adamax"
			case 5:
				optimizer = optim.ASGD(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "ASGD"
			case 6:
				optimizer = optim.NAdam(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "NAdam"
			case 7:
				optimizer = optim.RAdam(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "RAdam"
			case 8:
				optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, alpha=0.90)
				name = "RMSprop"
			case 9:
				optimizer = optim.Rprop(model.parameters(), lr=lr)
				name = "Rprop"
			case 10:
				optimizer = optim.SGD(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "SGD"
			case 11:
				optimizer = Lion(model.parameters(), lr=lr/20, weight_decay=wd)
				name = "Lion"
		
		for i in range(N): 
			theta = torch.rand(batch_size) * math.pi * 2.0
			x = torch.stack((torch.sin(theta), torch.cos(theta)), 0)
			# x = torch.randn(2, batch_size) 
			# ^ this does not change the results; just makes them slightly noisier
			x = x.T
			x = x.unsqueeze(1)
			y = x @ m
			p = model(x)
			loss = torch.sum((y - p)**2)
			loss.backward()
			optimizer.step()
			losses[k,i] = loss.detach()
			
	# do the same for linear regression
	theta = torch.rand(batch_size) * math.pi * 2.0
	x = np.stack((np.sin(theta), np.cos(theta)), 0)
	m_ = m.numpy() # re use the matrix
	x = x.T
	y = x @ m_
	mp = np.linalg.lstsq(x, y, rcond=None)[0]
	p = x @ mp
	loss = np.sum((y - p)**2)
	print(f"loss on one batch using linear regression {loss}")
	return losses, name

if __name__ == '__main__':
	plot_rows = 3
	plot_cols = 4
	figsize = (18, 15)
	fig, ax = plt.subplots(plot_rows, plot_cols, figsize=figsize)
	N = 4000
	phi = np.random.rand(5) * math.pi * 2.0
	for algo in range(12): 
	#for algo in [1,]:
		losses,name = main(phi, N, algo)
		r = algo // 4
		c = algo % 4
		ax[r,c].plot(losses.T)
		ax[r,c].set_title(name)
		
	plt.show()
