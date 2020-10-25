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
