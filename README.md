# Can you learn a 2x2 rotation matrix using Pytorch optimizers?

This is a afternoon experiment to answer the (apparently non-obvious!) question: can you use Pytorch optimizers to learn a 2x2 rotation matrix from a series of examples?  That is, given 
```math
m = \begin{matrix}cos(\phi) & -sin(\phi) \\ sin(\phi) & cos(\phi) \end{matrix}
y = m * x
```
Find m' from (x,y) supervised pairs.  

This is interesting because the direction of the gradients lie uniformly on a circle, and so stochastic optimizers seem to struggle!

![](plot.png)

Above, batch number is on the x-axis.  Batch size is fixed at 16, learning rate is 0.01 for Adadelta, Adagrad, RMSprop & Rprop, 0.0005 for the remainder to try to make them more stable.  Weight decay is set to 0.01 for those algorithms that permit it (though setting it to 0 did not change anything.)  Replicates are plotted with different colors (same m matrix for each color).  See the source for more details.

The network being trained is: 
```
class Net(nn.Module): 
	def __init__(self):
		super(Net, self).__init__()
		self.l1 = nn.Linear(2, 2, bias = True)
		torch.nn.init.zeros_(self.l1.weight)
		torch.nn.init.zeros_(self.l1.bias)
		
	def forward(self, x): 
		x = self.l1(x)
		return x
```
Note zero-initialization to make the process more repeatable (the results are mererly more noisy without it.)

Meanwhile, linear regression recovers the original matrix within eps in one batch. (Of course)

Suggested takeaways: even though everyone likes to use Adam for their transformers, it's unstable on this toy problem, for which Adagrad and RMSprop work, albiet with very poor data efficiency.

**Please submit a pull request if you find a bug!** I think this all is sound, exp given the presence of positive controls (RMSprop and linear regression), but I might be wrong!
