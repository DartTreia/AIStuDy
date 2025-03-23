import torch
import random

a= torch.randint(-10,10,(1,3))
print(a)

a = a.float()
print(a)

a.requires_grad=True  

b=a**2
print(b)

c=b*random.randint(1,10)
print(c)

d=c.exp()
print(d)



out = d.mean()
out.backward()
print(a.grad)
