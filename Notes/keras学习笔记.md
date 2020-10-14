2020.10.13

项目地址：[deep-learning-with-keras-notebooks](https://github.com/erhwenkuo/deep-learning-with-keras-notebooks)

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
