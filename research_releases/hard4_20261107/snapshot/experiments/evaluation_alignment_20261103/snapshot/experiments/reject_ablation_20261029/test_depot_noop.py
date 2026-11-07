import torch
from depot_noop import suppress_depot_noop
x=torch.tensor([[3.,1.,-torch.inf],[3.,1.,-torch.inf],[0.,-torch.inf,-torch.inf],[2.,-torch.inf,1.]],requires_grad=True)
p=x.log_softmax(-1).unsqueeze(1)
y=suppress_depot_noop(p,torch.tensor([True,False,True,True]))
assert torch.isneginf(y[0,0,0]) and y[0,0,1]==0
assert torch.allclose(y[1],p[1]) # return to depot from route remains legal
assert torch.allclose(y[2],p[2]) # terminal depot-only fallback retained
assert y[3,0,2]==0 # explicit rejection remains possible
assert torch.isneginf(y[p.isneginf()]).all()
(y.exp()*torch.arange(3)).sum().backward()
assert torch.isfinite(x.grad).all()
print('DEPOT_NOOP_TEST_PASS')
