2020.10.15

准备学习pytorch，先看一看要学习的内容，准备好软件环境

学习项目：[eat_pytorch_in_20_days](https://github.com/lyhue1991/eat_pytorch_in_20_days)


#### 10.23

学习进度：1-1 结构化数据建模流程范例

学习随记：

使用Pytorch实现神经网络模型的一般流程包括：

    1，准备数据

    2，定义模型

    3，训练模型

    4，评估模型

    5，使用模型

    6，保存模型。

#### 10.24

学习进度：1-2 图片数据建模流程范例

学习随记：

使用Pytorch通常有三种方式构建模型：使用nn.Sequential按层顺序构建模型，继承nn.Module基类构建自定义模型，继承nn.Module基类构建模型并辅助应用模型容器(nn.Sequential,nn.ModuleList,nn.ModuleDict)进行封装。

有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。

介绍了建构模型的流程

#### 10.25

学习进度：1-3 文本数据建模流程范例

学习随记：看看网络架构是如何构建的

还有就是有关画图的内容

#### 10.26

学习进度：第四节是有关rnn的，我就复习一下之前的内容，架构一下代码

#### 10.28

学习进度：pytorch核心概念  	2-1 张量数据结构

学习随记：

* 1.张量的数据结构

张量的数据类型和numpy.array基本一一对应，但是不支持str类型。

* 2.张量的维度

不同类型的数据可以用不同维度(dimension)的张量来表示。
    
    标量为0维张量，向量为1维张量，矩阵为2维张量。

彩色图像有rgb三个通道，可以表示为3维张量。视频还有时间维，可以表示为4维张量。可以简单地总结为：有几层中括号，就是多少维的张量。

* 3.张量的尺寸

可以使用 shape属性或者 size()方法查看张量在每一维的长度.

可以使用view方法改变张量的尺寸。

如果view方法改变尺寸失败，可以使用reshape方法.

* 4.张量和numpy数组

可以用numpy方法从Tensor得到numpy数组，也可以用torch.from_numpy从numpy数组得到Tensor。这两种方法关联的Tensor和numpy数组是共享数据内存的。

如果改变其中一个，另外一个的值也会发生改变。如果有需要，可以用张量的clone方法拷贝张量，中断这种关联。

此外，还可以使用item方法从标量张量得到对应的Python数值。使用tolist方法从张量得到对应的Python数值列表。

#### 10.29

学习进度：2-2 自动微分机制

学习随记：

* 1.利用backward方法求导数

backward 方法通常在一个标量张量上调用，该方法求得的梯度将存在对应自变量张量的grad属性下。如果调用的张量非标量，则要传入一个和它同形状 的gradient参数张量。

相当于用该gradient参数张量与调用张量作向量点乘，得到的标量结果再反向传播。

    1.标量的反向传播
    
        y.backward()
        dy_dx = x.grad

    2.非标量的反向传播
    
    3.非标量的反向传播可以用标量的反向传播实现

* 2.利用autograd.grad方法求导数

        dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]

* 3.利用自动微分和优化器求最小值
    
        optimizer = torch.optim.SGD(params=[x],lr = 0.01)
        
        def f(x):
            result = a*torch.pow(x,2) + b*x + c 
            return(result)
        
        for i in range(500):
            optimizer.zero_grad()
            y = f(x)
            y.backward()
            optimizer.step()
            
#### 10.30

学习进度：温习

学习随记：其实前几天都没太好好看项目的内容，今天开始好好再从头看看，然后看看有没有时间往下接着看

补充一下前面的笔记内容

#### 10.31

学习进度：先复习，再看看有没有时间再往下看看

2-3 动态计算图

学习随记：利用jupyter notebook调试下程序，好好学习一下

* 1.简介

Pytorch的计算图由节点和边组成，节点表示张量或者Function，边表示张量和Function之间的依赖关系。

Pytorch中的计算图是动态图。这里的动态主要有两重含义：

        第一层含义是：计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。

        第二层含义是：计算图在反向传播后立即销毁。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，或者利用torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。

1.计算图的正向传播是立即执行的

2.计算图在反向传播后立即销毁。

* 2.计算图中的Function

计算图中的 张量我们已经比较熟悉了, 计算图中的另外一种节点是Function, 实际上就是 Pytorch中各种对张量操作的函数。
这些Function和我们Python中的函数有一个较大的区别，那就是它同时包括正向计算逻辑和反向传播的逻辑。
我们可以通过继承torch.autograd.Function来创建这种支持反向传播的Function

* 3.计算图与反向传播

        import torch 

        x = torch.tensor(3.0,requires_grad=True)
        y1 = x + 1
        y2 = 2*x
        loss = (y1-y2)**2

        loss.backward()

loss.backward()语句调用后，依次发生以下计算过程。

    1，loss自己的grad梯度赋值为1，即对自身的梯度为1。

    2，loss根据其自身梯度以及关联的backward方法，计算出其对应的自变量即y1和y2的梯度，将该值赋值到y1.grad和y2.grad。

    3，y2和y1根据其自身梯度以及关联的backward方法, 分别计算出其对应的自变量x的梯度，x.grad将其收到的多个梯度值累加。

    （注意，1,2,3步骤的求梯度顺序和对多个梯度值的累加规则恰好是求导链式法则的程序表述）

正因为求导链式法则衍生的梯度累加规则，张量的grad梯度不会自动清零，在需要的时候需要手动置零。

* 4.叶子节点和非叶子节点

在反向传播过程中，只有 is_leaf=True 的叶子节点，需要求导的张量的导数结果才会被最后保留下来。

那么什么是叶子节点张量呢？叶子节点张量需要满足两个条件。

    1，叶子节点张量是由用户直接创建的张量，而非由某个Function通过计算得到的张量。

    2，叶子节点张量的 requires_grad属性必须为True.

Pytorch设计这样的规则主要是为了节约内存或者显存空间，因为几乎所有的时候，用户只会关心他自己直接创建的张量的梯度。

所有依赖于叶子节点张量的张量, 其requires_grad 属性必定是True的，但其梯度值只在计算过程中被用到，不会最终存储到grad属性中。

如果需要保留中间计算结果的梯度到grad属性中，可以使用 retain_grad方法。 如果仅仅是为了调试代码查看梯度值，可以利用register_hook打印日志。

* 5.计算图在TensorBoard中的可视化

#### 11.1

学习进度：接着昨天看看动态计算图，笔记补在昨天的内容上了

准备看看第三章 Pytorch的层次结构

Pytorch的层次结构从低到高可以分成如下五层。

    最底层为硬件层，Pytorch支持CPU、GPU加入计算资源池。

    第二层为C++实现的内核。

    第三层为Python实现的操作符，提供了封装C++内核的低级API指令，主要包括各种张量操作算子、自动微分、变量管理. 如torch.tensor,torch.cat,torch.autograd.grad,nn.Module. 如果把模型比作一个房子，那么第三层API就是【模型之砖】。

    第四层为Python实现的模型组件，对低级API进行了函数封装，主要包括各种模型层，损失函数，优化器，数据管道等等。 如torch.nn.Linear,torch.nn.BCE,torch.optim.Adam,torch.utils.data.DataLoader. 如果把模型比作一个房子，那么第四层API就是【模型之墙】。

    第五层为Python实现的模型接口。Pytorch没有官方的高阶API。为了便于训练模型，作者仿照keras中的模型接口，使用了不到300行代码，封装了Pytorch的高阶模型接口torchkeras.Model。如果把模型比作一个房子，那么第五层API就是模型本身，即【模型之屋】。
    
#### 11.2

学习进度：3-1,低阶API示范

学习随记：

低阶API主要包括张量操作，计算图和自动微分。

* 1.线性回归模型

1.准备数据

2.定义模型

3.训练模型

* 2.DNN二分类模型

1.准备数据

2.定义模型

3.训练模型

#### 11.4

这两天修改文章，可能要再延后几天看看学习项目，之后再补充下笔记

补充一下笔记，再看看线性回归模型的构建，看看是怎么构建起来的

#### 11.5

学习进度：3-2 中阶API示范

学习随记：Pytorch的中阶API主要包括各种模型层，损失函数，优化器，数据管道等等。

* 1.线性回归模型


#### 11.7   11.8

这两天忙着改文章，先暂停一下

#### 11.9

还在改文章，加油！

#### 11.10

待更新

#### 11.11

这几天忙着修改文章，之前的进度都落下了，先回顾一下，然后补充一下之前的笔记

学习进度：3-2 中阶API示范

然后找个项目看看代码进一步熟悉一下应用

#### 11.12

静下心来，继续好好学习一下pytorch

#### 11.13

重新看看，把之前的笔记都补充一下

低阶API主要包括张量操作，计算图和自动微分。

Pytorch的中阶API主要包括各种模型层，损失函数，优化器，数据管道等等。

Pytorch没有官方的高阶API，一般需要用户自己实现训练循环、验证循环、和预测循环。通过仿照tf.keras.Model的功能对Pytorch的nn.Module进行了封装，实现了 fit, validate，predict, summary 方法，相当于用户自定义高阶API。

学习随记：具体看看第四章 低阶api的具体介绍

* 张量的结构操作

张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

张量数学运算主要有：标量运算，向量运算，矩阵运算。张量运算的广播机制。

* 索引切片

张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。可以通过索引和切片对部分元素进行修改。

此外，对于不规则的切片提取,可以使用torch.index_select, torch.masked_select, torch.take

对于不规则的切片提取,可以使用torch.index_select, torch.take, torch.gather, torch.masked_select.

如果要通过修改张量的某些元素得到新的张量，可以使用torch.where,torch.masked_fill,torch.index_fill

        torch.where可以理解为if的张量版本。

        torch.index_fill的选取元素逻辑和torch.index_select相同。

        torch.masked_fill的选取元素逻辑和torch.masked_select相同。

* 维度变换

维度变换相关函数主要有 torch.reshape(或者调用张量的view方法), torch.squeeze, torch.unsqueeze, torch.transpose

        torch.reshape 可以改变张量的形状。

        torch.squeeze 可以减少维度。

        torch.unsqueeze 可以增加维度。

        torch.transpose 可以交换维度。如果是二维的矩阵，通常会调用矩阵的转置方法 matrix.t()，等价于 torch.transpose(matrix,0,1)。

* 合并分割

可以用torch.cat方法和torch.stack方法将多个张量合并，可以用torch.split方法把一个张量分割成多个张量。

torch.cat和torch.stack有略微的区别，torch.cat是连接，不会增加维度，而torch.stack是堆叠，会增加维度。

torch.split是torch.cat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。

#### 11.14

接着补充一下笔记，看看之前的内容，主要是张量的操作，正看完tf那边的内容，再看看pytorch的，结合起来学习

#### 11.15

补充一下笔记，接着看看下面的章节

学习进度：4.2 张量的数学运算

#### 11.18

这几天都没怎么看pytorch的内容，准备这些天抓紧时间看看，下个月开始上手相关的项目，加油！

学习进度：第四章的内容，补充一下之前的笔记

#### 11.19

学习进度：看一下 5 Pytorch的中阶API  的内容

复习一下昨天的第四章

#### 11.21

结合着keras的学习一起看看相对应的章节，综合学习

#### 12.1

开始好好研究pytorch了，从开源的项目入手，进行试验

回顾一下学习的内容，也学习keras，侧重点在pytorch

#### 12.2

从头看看pytorch

#### 12.4

这几天有点儿忽略pytorch了，从yolov3的项目入手吧，分析分析代码

#### 12.6

继续看看pytroch，然后可以的话跑跑项目

#### 12.25

好好学习pytorch了，把之前落下的捡起来
