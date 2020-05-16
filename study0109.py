import numpy as np
#import pandas as pd
import os



#annotation_path = 'test.txt'
#with open(annotation_path) as f:
    #annotation_lines = f.readlines()

#line = annotation_lines[0].split()
#print(line)

#file_path = '/'.join(line[0].split('/')[-4:])
#print(file_path)

#image_file = file_path.split('/')[-1].split('.')[0]
#print(image_file)

#box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
#print(box)


# np_data = np.random.randint(0, 6, [4,2,3,4])
# print(np_data)
#result:
# [[[[1 3 4 0]   #D0000-D0003
#    [4 1 1 4]   #D0010-D0013
#    [2 3 2 3]]  #D0020-D0023
#
#   [[0 4 2 5]   #D0100
#    [5 4 1 5]
#    [5 4 0 4]]]        #D0123
#
#
#  [[[5 5 4 4]  #D1000
#    [3 2 5 3]
#    [4 0 3 4]]
#
#   [[3 0 3 2]   #D1100
#    [5 2 4 1]
#    [4 0 4 5]]]
#
#
#  [[[3 2 0 2]   #D2000
#    [5 0 4 0]
#    [2 0 1 2]]
#
#   [[0 4 0 5]   #D2100
#    [2 2 0 2]
#    [5 2 0 5]]]
#
#
#  [[[2 2 1 3]   #D3000
#    [0 4 0 2]
#    [5 3 0 5]]
#
#   [[1 4 4 1]   #D3100
#    [2 1 2 1]
#    [1 5 4 2]]]]

# a = np.array(np.arange(12).reshape(3,4))
# print(a)
# a_tum=np.insert(a,0,-1.0)
# print(a_tum)

a  = np.logspace(-2.0, 0.0, num = 9)
print(a)

from keras.layers import ZeroPadding2D
x = ZeroPadding2D(((1,0),(1,0)))
