
import torch
import numpy as np
from PIL import Image
import math

"""
x = torch.rand(512, 512, 3)
print(x.size())
x = x.to("cpu").detach().numpy().copy()

def f(i, j, k):
	i = i * 0.1
	j = j * 0.1
	k = k * 0.1
	rand = math.sin(i * 0.1) + 2.0 * math.sin(1.2 * j + k) + 2.1 * math.cos(k)
	rand += 2.5
	return (int)(rand * 51.0)

for i in range(512):
	for j in range(512):
		for k in range(3):
			x[i][j][k] = f(i, j, k)
"""


"""
a = np.zeros((2, 2))
print(a)
a[1][0] = 1.0

print(np.amax(a))
print(np.amin(a))

pil_img = Image.fromarray(x.astype(np.uint8))
pil_img.show()
"""

x = torch.ones(3, 3)
print(x)

y = torch.zeros(3, 3)
print(y)

print(torch.cat((x, y), 1))
print(torch.cat([x, y], 1))