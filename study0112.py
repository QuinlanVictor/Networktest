from __future__ import print_function
import torch

# x= torch.tensor([5.5, 3])
# print(x)
#
# x=x.new_ones(5,3,dtype=torch.double)
# print(x)
#
# x=torch.randn_like(x,dtype=torch.float)
# print(x)
#
# print(x.size())  #返回的是元组tuple属性
#
# #加法
# y=torch.rand(5,3)
# print(y+x)
#
# print(torch.add(x, y))

#torch.view与numpy和reshape类似
# x=torch.randn(4,4)
# y=x.view(16)
# z=x.view(-1,8)
# print(x.size(),y.size(),z.size())




#0114keras学习
import numpy as np
input_shape=(416,416)
input_shape = np.array(input_shape, dtype='int32')
# y=input_shape[::-1]
# print(y)

num_layers=3
grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
# g=input_shape//{0:32, 1:16, 2:8}[1]
m=6
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
num_classes=1
y_true = [np.zeros((m,grid_shapes[1][0],grid_shapes[1][1],len(anchor_mask[1]),5+num_classes),
        dtype='float32')]
print(y_true)
