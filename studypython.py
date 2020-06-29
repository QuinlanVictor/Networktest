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
