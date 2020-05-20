import numpy as np

#7 创建一个值从10到49的数组
z=np.arange(10,50)

#8 反转数组
z=np.arange(50)
z=z[::-1]

#9 创建一个从0~8的3*3矩阵
Z=np.arange(9).reshape(3,3)  #arange函数用于创建等差数组
print(Z)

#10 从[1,2,0,0,4,0]中找到非0元素的索引
nz = np.nonzero([1,2,0,0,4,0]) #找到非0元素的索引
print(nz)

#11 生成一个3*3的对角矩阵
z1=np.eye(3)
print(z1)

#12 创建一个3*3*3的随机值数组
z2=np.random.random((3,3,3))

#13 创建一个10*10的随机值数组，并找到最大最小值
Z2 = np.random.random((10,10))
Z2min, Z2max = Z2.min(), Z2.max()
print(Z2min, Z2max)

#14 找到平均值
z3=np.random.random(30)
m=z3.mean()
print(m)

#15 创建一个四边为1，中间为0的二维数组
z4=np.ones((10,10))
z4[1:-1,1:-1]=0
print(z4)

#16 给一个已经存在的数组添加边（填充0）
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)

#17 查看下列式子表达的意思
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

#18 创建一个5*5矩阵，对角线下方值为1，2，3，4
Z = np.diag(1+np.arange(4),k=-1)
print(Z)
#array是一个一维数组时，结果形成一个以一维数组为对角线元素的矩阵
#array是一个二维矩阵时，结果输出矩阵的对角线元素
#k大于零对角线上面，小于零对角线下面

#19 创建一个8*8矩阵，并用棋盘图案填充
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1  #奇数行跳着来
Z[::2,1::2] = 1
print(Z.shape)

#print(np.unravel_index(99,(6,7,8)))  #给定一个678的三维矩阵，求100个元素的索引是什么？
#函数的作用是获取一个int类型的索引值在一个多维数组中的位置。

#20 给定一个678的三维矩阵，求100个元素的索引是什么？
print(np.unravel_index(99,(6,7,8)))

#21 使用tile函数创建8*8的棋盘矩阵
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)
#tile函数的主要功能就是将一个数组重复一定次数形成一个新的数组,但是无论如何,最后形成的一定还是一个数组

#22对一个5*5矩阵标准化处理
z5=np.random.random((5,5))
z5=(z5-np.mean(z5))/(np.std(z5)) #std是标准差，var方差，mean均值

#23新建一个dtype类型用来描述一个颜色（RGBA）
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])

#24 5*3矩阵和3*2矩阵相乘
z6=np.dot(np.ones((5,3)),np.ones((3,2)))


#25  给定一个一维数组，将第3~8个元素取反
z=np.arange(11)
z[(3<z)&(z<=8)] *= -1
print(z)


#26  看看下面脚本的输出是什么？
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


#28  下面表达式的结果是什么？
print(np.array(0) / np.array(0))
print(np.array(0) // np.array(0))
print(np.array([np.nan]).astype(int).astype(float))

#29  对数组进行四舍五入操作
z=np.random.uniform(-10,+10,10)
# numpy.random.uniform(low,high,size)   从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
print(np.copysign(np.ceil(np.abs(z)),z))
#abs计算绝对值，ceil计算大于等于该值的最小整数，copysign好像是交换两个array的标号，这里应该就是和z交换，恢复正负值因为之前做过了绝对值
#np.ceil和np.floor ceil向上取整，floor向下取整


#30  找出两个数组的共同值
z7=np.random.randint(0,10,10)
z8-np.random.randint(0,10,10)
print(np.intersect1d(z7,z8))
#np.random.randint 函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
#就是low，high，size 从0到10返回10个整数，不包括10


#31  忽略所有numpy警告
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

#An equivalent way, with a context manager:
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0


#32 下面表达式正确吗？
np.sqrt(-1) == np.emath.sqrt(-1)


#33 获得昨天、今天、明天的日期
yesterday=np.datetime64('today','D')-np.timedelta64(1,'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

#34 获得2016年7月对应的所有日期
z=np.arange('2016-07','2016-08',dtype='datetime64[D]')
print(z)


#35  计算((A+B)*(-A/2))
A = np.ones(3)*1
B = np.ones(3)*2
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)


#36  提取随机数列整数部分的五种方法
Z = np.random.uniform(0,10,10)
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))  #截取整数部分

#37 创建一个5*5的矩阵，每一行值为1~4
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

#38 给定一个生成器函数，可以生成10个整数，使用它来创建一个数组
def generate():
    for x in range(10):
        yield x
z=np.fromiter(generate(),dtype=float,count=-1)  #从可迭代对象创建新的1维数组
print(z)

#39  创建一个长度为10的数组，值为0~1之间，不包含首尾
z=np.linspace(0,1,11,endpoint=False)[1:]
print(z)
#linspace创建等差数列，false不包含11


#40  创建一个长度为10的数组，并做排序操作
z=np.random.random(10)
z.sort()
print(z)

#41 对一个数组进行相加操作，并且速度快于np.sum
z=np.arange(10)
np.add.reduce(z)
#对于相对较小的数组，add.reduce大约快两倍

#42  给定两个随机数组A和B，验证它们是否相等
A=np.random.randint(0,2,5)
B=np.random.randint(0,2,5)

equal=np.allclose(A,B)  #比较两个array是不是每一元素都相等
print(equal)

#43 使一个数组不变（只读）
z=np.zeros(10)
z.flags.writable = False
z[0]=1

#44 给定表示笛卡尔坐标的一个10*2的随机矩阵，将其转换为极坐标
z-np.random.random((10.2))
X,Y=z[:,0],z[:,1]
r=np.sqrt(x**2+y**2)
t=np.arctan2(Y,X)
print(r)
print(t)


#45  创建一个长度为10的随机矩阵，并将最大值替换为0
z=np.random.random(10)
z[z.argmax()]=0  #返回最大值的索引
print(z)

#46 创建具有x和y坐标的结构化数组，它们覆盖[0,1] x [0,1]区域
z=np.zeros((5,5),[('x',float),('y',float)])
z['x'],z['y']=np.meshgrid(np.linspace(0,1,5),
                          np.linspace(0,1,5))
#np.meshgrid生成网格点坐标矩阵

#47 给定两个数组X和Y，构造柯西矩阵C（Cij = 1 /（xi-yj））
x=np.arange(8)
y=x+0.5
c=1.0 / np.subtract.outer(x,y)
print(np.linalg.det(c))
#np.subtract.outer xy中每一个元素进行比较
#np.linalg.det 矩阵求行列式


#48 打印每种numpy标量类型的最小和最大可表示值
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

#np.iinfo 取数作为临时极大值、极小值的方法

#49  打印数组中所有值
np.set_printoptions(threshold=np.nan) #设置输出样式。这里的意思是全部打印
Z = np.zeros((16,16))
print(Z)

#50 在数组中找到最接近给定值的值
z=np.arange(100)
v=np.random.uniform(0,100)
index=(np.abs(z-v)).argmin()
print(z[index])

