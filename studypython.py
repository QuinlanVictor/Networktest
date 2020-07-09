#学习python随记
class person:
    hair='black'
    def __init__(self,name='Charlie',age=8):
        self.name=name
        self.age=age
    def say(self,content):
        print(content)

p=persom

class dog:
    def junmp(self):
        print(",,,")
    def run(self):
        self.junmp()
        print("..")

#全局空间
class user:
    def test(self):
        print('dd',self)

u=user
u.test()
foo=u.test()
foo()

def foo():
    print()
bar=20
class bird:
    def foo():#定义bird空间的foo函数
        print("")
    bar=200
foo()
print(bar)
bird.foo()
print(bird.bar)

#类方法和静态方法
class bird:
    @classmethod
    def fly(cls):
        print("")
    @staticmethod
    def info(p):
        print("",p)

bird.fly() #bird类会自动绑定到第一个参数
bird.info('crazyit')#静态方法不会自动绑定
b=bird()
b.fly()
b.info('ooo')


#函数修饰器
def funa(fn):
    print("")
    fn()
    return 'fkit'

@funa #相当于funa（funb）
def funb():
    print('b')
print(funb)

#命名空间
class item:
    print('')
    for i in range(10):
        if i%2==0:
            print('',i)
        else:
            print('',i)


#异常处理机制

#变量

#列表

#第三章
list()
tuple()



#第八章
#特殊方法
#常见的特殊方法

#第九章


#k近邻算法

#模块和包
#sys  os   random   time  json支持  正则表达式

#第十一章 图形界面编程
#GUI库
#创建一个简单窗口
from tkinter import *
root=Tk()
root.title('')
w=Label(root,text='')
w.pack()
root.mainloop()

#创建一个窗口，使用循环创建三个Label
from tkinter import *
root=Tk()
root.title('PACK')
for i in range(3):
    lab=Label(root,text='',bg='')
    lab.pack()
root.mainloop()

#多个容器（Frame）进行布局
from tkinter import *
class App:
    def __init__(self,master):
        self.master=master
        self.initWidgets()
    def initWidgets(self):
        fm1=Frame(self.master)
        fm1.pack(side=LEFT,fill=BOTH,expand=YES)


#文件I/O
from pathlib import *
pp=PurePath('')
pp=PurePath('','','')
pp=PurePath(Path(''),Path(''))

#匹配
from pathlib import *
import fnmatch
for file in Path('.').iterdir():
    if fnmatch.fnmatch(file,''):
        print(file)
