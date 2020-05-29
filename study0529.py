import numpy as np

x=img.reshape((32*32*,1))

c=np.dot(a,b)

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a * b
#广播机制，b会被复制三次


