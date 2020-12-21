2020.10.13

项目地址：[deep-learning-with-keras-notebooks](https://github.com/erhwenkuo/deep-learning-with-keras-notebooks)

1016note：可以配合着书籍《Python深度学习》一书看看，这本书算是keras入门书籍，内容也与项目中的笔记部分一致，因为笔记使用的是繁体，一些翻译习惯也不太一样，所以参照书本可能更直观一些

正好之前也在重新复习这本书，正好作为一个学习随记


    实验环境
    import platform
    import tensorflow
    import keras
    print("Platform: {}".format(platform.platform()))
    print("Tensorflow version: {}".format(tensorflow.__version__))
    print("Keras version: {}".format(keras.__version__))

#### 10.13  

学习进度 ： keras API范例 1.3 有关预训练模型的内容

学习随记：


#### 10.14

学习进度 ： keras API范例 1.4 图像增强训练小数据集

学习随记：回顾一下昨天的学习内容

jpg文件的解码过程：

        读取图片
        
        将jpg内容解码为rgb像素
        
        将其转换为浮点数计算
        
        将像素值重新缩放到（0,1）之间
 
 代码：
 
        from keras.preprocessing.image import ImageDataGenerator
        #将图像文件转换为张量tensor
        x_train = ImageDataGenerator(rescale=1./255)  #例子：归一化处理
        x_generator = x_train.flow_from_directory(
            x_train,
            target_size=(150,150),
            batch_size=20,
            class_mode='binary'

        )
        #使用flow_from_directory制作自己的数据集，databatch shape:(20,150,150,3),label shape :(20,)


#### 10.15

学习进度：keras API范例 1.5 使用预训练的网络模型

学习随记：回顾一下之前的内容

使用预训练的网络结构：1是直接运行网络然后将得到的输出，输入到独立的分类网络上
                    2是在输入的部分加入新的结构，扩充数据集，冻结网络结构进行训练

微调


#### 10.16

学习进度：keras API范例 1.6 可视化

学习随记：回顾之前的学习内容

中间激活的可视化：

        from keras.models import load_model
        model = load_model('')
        model.summary()

        img_path = ''

        from keras.preprocessing import image
        import numpy as np
        img = image.load_img('')
        img_tensor = image.img_to_array(img)

        img_tensor = np.expand_dims(img_tensor,axis=0)
        img_tensor /= 255.


        from keras import models
        layer_outputs = [layer.output for layer in model.layers[:8]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(img_tensor)
        first_layer_activation = activations[0]
        print(first_layer_activation.shape) # 看一下第一層產生的shape

可视化卷积神经网络的过滤器：


可视化类激活的热力图：



#### 10.17

学习进度： keras API范例 1.7 Autoencoder

学习随记：


#### 10.18

学习进度：keras API范例 1.8 1.9


#### 10.19

学习进度：keras API范例 1.8 1.9
         
         图像识别 2.1
 
学习随记：接着昨天的学习内容，昨天基本上就是简单看了看，并没有太深入的学习

先好好复习，看看内容

记住一个图像增强可以用到的： ImageDataGenerator

    from keras.preprocessing.image import ImageDataGenerator


#### 10.20

学习进度：图像识别 2.2

#### 10.21

学习进度：图像识别 2.3 2.4 2.5

#### 10.22

学习进度：看看之前学习的内容

#### 10.23

学习进度：物体侦测 3.1  3.2 

学习随记：介绍了物体检测的一些方法起源，3.1介绍了YOLO的思想

[YAD2K](https://github.com/allanzelener/YAD2K) YOLO V2实现的项目，90%使用keras编写，10%使用TensorFlow

图像张量需要进行归一化处理（除以255）和数据类型为浮点数32

之前加入的打印网络结构的代码我又忘记加入训练代码中了，有时间记得再跑一次，还没看过效果

#### 10.24

学习进度：物体侦测 3.3 3.4

#### 10.25

学习进度：物体侦测 3.5 3.6 3.7

学习随记：怎么第四五六章的内容没了，纳闷，之前都没发现

#### 10.26

学习进度：看看之前的内容，然后接着看看后面的章节

学习随记：考虑下是不是换个项目再接着看看学习一下

#### 10.27

学习进度：Face Detection/Recognition 7.0

学习随记：找找其他学习资源

#### 10.28

学习进度：了解下自然语言处理

#### 10.30

学习进度：温习一下之前的内容

#### 10.31

学习进度：复习之前的学习内容

学习随记：昨天调试好了jupyter notebook，在上面好好测试一下keras的相关程序，多练练代码

#### 11.2

学习进度：复习一下之前的内容，看看yolov3的改进，思考一下

#### 11.4

这两天修改文章，可能要再延后几天看看学习项目，之后再补充下笔记

#### 11.5

keras这边暂时没什么学习进展，最近在修改网络结果，尝试新的网络，看看还有哪些需要再学的

#### 11.7   11.8

这两天忙着改文章，先暂停一下

#### 11.9

还在改文章，加油！

#### 11.10

待更新

#### 11.12

学习项目：[tf学习资料](https://github.com/lyhue1991/eat_tensorflow2_in_30_days)

同时参考一下这个项目，进行一下学习

学习随记：

使用Keras接口有以下3种方式构建模型：使用Sequential按层顺序构建模型，使用函数式API构建任意结构模型，继承Model基类构建自定义模型。

训练模型通常有3种方法，内置fit方法，内置train_on_batch方法，以及自定义训练循环。

Tensorflow底层最核心的概念是张量，计算图以及自动微分。

###### 有关计算图的问题要好好看看

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph

#### 11.13

学习进度：接着学习，补充一下笔记

自动微分机制：神经网络通常依赖反向传播求梯度来更新网络参数，求梯度过程通常是一件非常复杂而容易出错的事情。Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值。

#### 11.14

学习进度：接着昨天的内容，补充一下笔记.主要看看一些具体的操作。主要去看看有关张量的操作

* 张量的结构操作

张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。

张量数学运算主要有：标量运算，向量运算，矩阵运算。张量运算的广播机制。

###### 创建张量

###### 索引切片

张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。对于tf.Variable,可以通过索引和切片对部分元素进行修改。

对于提取张量的连续子区域，也可以使用tf.slice.此外，对于不规则的切片提取,可以使用tf.gather,tf.gather_nd,tf.boolean_mask。

tf.boolean_mask功能最为强大，它可以实现tf.gather,tf.gather_nd的功能，并且tf.boolean_mask还可以实现布尔索引。

如果要通过修改张量的某些元素得到新的张量，可以使用tf.where，tf.scatter_nd。tf.where可以理解为if的张量版本，此外它还可以用于找到满足条件的所有元素的位置坐标。tf.scatter_nd的作用和tf.gather_nd有些相反，tf.gather_nd用于收集张量的给定位置的元素，而tf.scatter_nd可以将某些值插入到一个给定shape的全0的张量的指定位置处。

###### 维度变换

维度变换相关函数主要有 tf.reshape, tf.squeeze, tf.expand_dims, tf.transpose.

tf.reshape 可以改变张量的形状。tf.squeeze 可以减少维度。tf.expand_dims 可以增加维度。tf.transpose 可以交换维度。

        tf.reshape可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序，所以，该操作实际上非常迅速，并且是可逆的。

        如果张量在某个维度上只有一个元素，利用tf.squeeze可以消除这个维度。和tf.reshape相似，它本质上不会改变张量元素的存储顺序。张量的各个元素在内存中是线性存储的，其一般规律是，同一层级中的相邻元素的物理地址也相邻。
        
        tf.transpose可以交换张量的维度，与tf.reshape不同，它会改变张量元素的存储顺序。tf.transpose常用于图片存储格式的变换上。

###### 合并分割

和numpy类似，可以用tf.concat和tf.stack方法对多个张量进行合并，可以用tf.split方法把一个张量分割成多个张量。

tf.concat和tf.stack有略微的区别，tf.concat是连接，不会增加维度，而tf.stack是堆叠，会增加维度。

tf.split是tf.concat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。

#### 11.15

学习进度：接着昨天的内容再复习一下张量的操作，然后再往下去看看

学习tf也可以引用到keras中来，毕竟低版本有from keras import backend as K，高版本听说要用tf.keras了

* 张量的数学运算

###### 标量运算

张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。

标量运算符的特点是对张量实施逐元素运算。有些标量运算符对常用的数学运算符进行了重载。并且支持类似numpy的广播特性。许多标量运算符都在 tf.math模块下。

        # 幅值裁剪
        x = tf.constant([0.9,-0.8,100.0,-20.0,0.7])
        y = tf.clip_by_value(x,clip_value_min=-1,clip_value_max=1)
        z = tf.clip_by_norm(x,clip_norm = 3)

        [0.9 -0.8 1 -1 0.7]
        [0.0264732055 -0.0235317405 2.94146752 -0.588293493 0.0205902718]
        
###### 向量运算

向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。 许多向量运算符都以reduce开头。

axis=1 横轴   axis=0 竖轴

###### 矩阵运算

矩阵必须是二维的。类似tf.constant([1,2,3])这样的不是矩阵。

矩阵运算包括：矩阵乘法，矩阵转置，矩阵逆，矩阵求迹，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。

除了一些常用的运算外，大部分和矩阵有关的运算都在tf.linalg子包中。

###### 广播机制

TensorFlow的广播规则和numpy是一样的:

        1、如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样。
        2、如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的。
        3、如果两个张量在所有维度上都是相容的，它们就能使用广播。
        4、广播之后，每个维度的长度将取两个张量在该维度长度的较大值。
        5、在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。

tf.broadcast_to 以显式的方式按照广播机制扩展张量的维度。

________

随记：把这两章的内容再仔细看看，反复学习一下

#### 11.16

学习进度：4.3 AutoGraph的使用规范

学习随记：把之前张量的结构和运算这两章再好好看看，其中有关的内容值得复习几次。有一个问题是我看的这个项目主要针对的是tf2.0，所以可能有些不适合我所用发tf版本，需要考虑下这方面的问题

有三种计算图的构建方式：静态计算图，动态计算图，以及Autograph。TensorFlow 2.0主要使用的是动态计算图和Autograph。

动态计算图易于调试，编码效率较高，但执行效率偏低。静态计算图执行效率很高，但较难调试。

而Autograph机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利。当然Autograph机制能够转换的代码并不是没有任何约束的，有一些编码规范需要遵循，否则可能会转换失败或者不符合预期。

#### 11.17

学习进度：4.4 AutoGraph的机制原理

#### 11.18

学习进度：4.5 AutoGraph和tf.Module

复习复习之前的内容

#### 11.19

学习进度：5 TensorFlow的中阶API

把第五章的内容看看，对应的api相应的再复习一下，这都是平常能用到的

#### 11.20

学习进度：继续往下看看，然后复习一下之前学习的内容

重新看看第二章的内容:tf.Variable和tf.constant，一个是变量张量，可以训练的，一个是常量张量，不能修改

看到了with语句：with open(annotation_path) as f   和   with tf.GradientTape() as tape

Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值。这种利用tf.GradientTape求微分的方法叫做Tensorflow的自动微分机制。

如果需要在TensorFlow2.0中使用静态图，可以使用@tf.function装饰器将普通Python函数转换成对应的TensorFlow计算图构建代码。运行该函数就相当于在TensorFlow1.0中用Session执行代码。使用tf.function构建静态图的方式叫做 Autograph.

#### 11.21 

接着昨天的内容继续看看四五章

#### 11.22

把全部内容都看看吧，查漏补缺

#### 11.23

重新重点看看第四章有关张量的一些计算

#### 11.26

看看相关的张量运算

#### 11.27

疯狂复习国考中，先暂停两天

#### 11.28

明天考完归来

#### 11.29

继续研究keras的相关学习内容

#### 12.4

继续从张量那块入手，学习keras的深层次编程

#### 12.10

看看相关的一些项目，复现一下最新的进展

#### 12.21

终于有时间了，也开始看看keras的学习，加油

