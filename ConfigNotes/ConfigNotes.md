## 踩坑笔记

### 记录一下配置环境时遇到的问题

#### 11.25

把配置环境时遇到的一些问题记录在这里，有时间总结一下之前遇到的坑，先从今日遇到的问题开始记录。

遇到的是使用anaconda安装SimpleITK库的问题，直接安装好像总是不成功，我使用如下语句安装成功了：

        conda install -c simpleitk/label/dev simpleitk
        
这好像安装的是最新版的，附上一个github地址，[SimpleITK](https://github.com/SimpleITK/SimpleITK/releases)

再附上一个解决办法的博客 [CSDN Python SimpleITK库的安装](https://blog.csdn.net/weixin_44217573/article/details/108624916)

_______

然后是pydicom的安装

        conda install -c conda-forge pydicom

附上在github的地址，[pydicom](https://github.com/pydicom/pydicom)
