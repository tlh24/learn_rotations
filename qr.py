import torch
import pdb

def QRdecomp(A): 
	# pdb.set_trace()
	[m,n] = A.shape
	Q = torch.eye(m,n) # orthogonal transform so far
	R = A.clone() # transformed matrix so far
	
	for j in range(n): 
		# find H = I - tau * w * w' to put zeros below R(j,j)
		normx = torch.linalg.vector_norm(R[j:,j]) # len x
		s = -1. * torch.sign(R[j,j])
		if j == n-1: 
			s = s*-1 # ??? 
		u1 = R[j,j] - s*normx # R[j,j] + normx if R[j,j] is positive
		w = R[j:,j] / u1 # column vector
		w[0] = 1.0
		tau = -s*u1 / normx
		# tau*w = (-s*u1 / normx) * (R[j:,j] / u1)  -- u1 cancels
		# tau*w = (-s / normx) * R[j:,j] 
		#       = -s * R[j:,:] / norm(R[j:,j])
		
		# Householder transform is H = I - 2 v v' (v is a column vector)
		# below distributes R and Q
		R[j:,:] = R[j:,:] - torch.outer(tau*w, w) @ R[j:,:]
		Q[:,j:] = Q[:,j:] - Q[:,j:] @ torch.outer(w, tau*w)

	return Q,R

if __name__ == '__main__':
	A = torch.randint(0,10, (3,3), dtype=torch.float)
	Q,R = QRdecomp(A)
	qq,rr = torch.linalg.qr(A)
	print(Q)
	print(R)
	print(qq)
	print(rr)
	print(torch.sum(Q-qq))
	print(torch.sum(R-rr))
	# last column consistently the wrong sign - why?
