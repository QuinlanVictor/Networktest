date：20201023

发现pandas在数据处理中也用的比较多，找找相对应的学习项目学习一下

学习项目：[or-pandas](https://github.com/zhouyanasd/or-pandas)

学习书籍：《基于Python的大数据分析基础及实战》

##### 10.24

学习进度：  预备章：Jupyter简介   第一章：数据分析入门

##### 10.26

学习进度：  jupyter的使用

学习随记：

jupyter

  神奇的预定义功能：%lsmagic

  逐行方式是执行单行的命令，而逐单元方式则是执行不止一行的命令，而是执行整个单元中的整个代码块。

  在逐行方式中，所有给定的命令必须以 % 字符开头；而在逐单元方式中，所有的命令必须以 %% 开头。

pandas：

主要有两种数据结构：Series系列和Dataframe数据框

    Series是一种类似于以为NumPy数组的对象，它由一组数据（各种NumPy数据类型）和与之相关的一组数据标签（即索引）组成的。可以用index和values分别规定索引和值。如果不规定索引，会自动创建 0 到 N-1 索引。
    
    DataFrame是一种表格型结构，含有一组有序的列，每一列可以是不同的数据类型。既有行索引，又有列索引。

* 1.从具有索引标签的字典数据创建一个DataFrame df.

      df = pd.DataFrame(data,index = labels)

* 2.从numpy 数组构造DataFrame

      df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                    columns=['a', 'b', 'c'])

* 3.通过其他DataFrame来创建DataFrame df3

* 4.从csv文件中每隔n行来创建Dataframe

* 5.用Series创建DataFrame

##### 10.30

学习进度：回顾下之前的进度

学习随记：今天算是搞定了jyputer notebook的应用，怎么使用本地的虚拟环境

##### 10.31

学习进度：补充一下之前的笔记内容，看看有没有时间再往下接着学习

学习随记：  第二章：数据的导入与导出

##### 11.1

学习进度：第二章：数据的导入与导出

学习随记：

* 1.list、dict、np.array 格式数据

