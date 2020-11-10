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

list

    pd.DataFrame([1,2,3,4])

dict

    data = {'a':[1,2],'b':[2,3]}
    pd.DataFrame(data)
    # 或者
    pd.DataFrame.from_dict(data)

np.array

    #有一个txt数据文件
    data = np.loadtxt('fit.txt', delimiter=None, comments='%',  usecols=(0, 1, 4,5))

其他方式

    with open(path, "r") as load_f:
        l = f.readlines()

* 2.文本格式数据格式数据

CSV文件

导入CSV文件

    pandas.read_csv(filepath_or_buffer, sep=',', header='infer', names=None, indxe_col=None)
    
    filepath_or buffer: str, path object or file-like object。指定传入的文件路径，必须传入的参数。
    sep: str。指定分隔符，默认是逗号分隔符。
    header: int, list or int。指定行数用来作为列名。默认是如果没有传入names参数，则header=0，用第一行作为列名，否则header=None，以传入的names作为列名。另外如果传入的是list，例如[0,1,3]，  则是以第1、2、4这些行作为多级列名，且中间的行，第3行会被忽略，数据从第5行开始。
    names: array-like, optional。指定文件的列名。如果文件中没有标题行，建议传入此参数。
    index_col: int, str, or sequence of int / str, or False。指定文件的索引，默认为None。
    
导入ex1.csv

    df = pd.read_csv('examples/ex1.csv')    

设置sep和header参数，导入ex2.csv

    df2 = pd.read_csv('examples/ex2.csv',sep='|',header=None)

设置sep和names参数，此时header默认为None

    df3 = pd.read_csv('examples/ex2.csv',sep='|', names=['ID','name','age','city','message

对ex1.csv设置多级标题，将第1、2、4行作为标题，数据从第5行开始

    df4 = pd.read_csv('examples/ex1.csv',header=[0,1,3])

导入ex1.csv，指定索引为message一列

    df5 = pd.read_csv('examples/ex1.csv',index_col='ID')

导入ex1.csv，指定第1和2列作为多重索引

    df6 = pd.read_csv('examples/ex1.csv',index_col=[0,1])
_____
导出csv数据

    DataFrame.to_csv(path_or_buf, index=True, header=True, sep=',', encoding='utf-8')

    path_or_buf: str or file handle。指定保存文件路径，必须传入的参数，默认为None。
    index: bool。导出的csv是否包含索引，默认为True。
    header: bool or list of str。导出的csv是否包含标题行，默认为True。
    sep: str。指定导出的csv文件的分隔符，默认为逗号分隔符。
    encoding: str。指定导出的csv文件的编码，默认为utf-8。

    # 导出文件
    df.to_csv("output/out_ex1.csv",index=False)

##### 11.2

学习进度：补充下之前的笔记内容，看看有没有时间再往下学习一下

#### 11.4

这两天修改文章，可能要再延后几天看看学习项目，之后再补充下笔记

#### 11.5

学习进度：补充下之前的笔记内容

excel文件

#### 11.7   11.8

这两天忙着改文章，先暂停一下

#### 11.9

还在改文章，加油！

#### 11.10

待更新
