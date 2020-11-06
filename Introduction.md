# 垃圾识别与检测

## 1. 简介
&emsp;基本原理： Mask R-CNN[<sup>1</sup>](#refer-anchor-1)  
&emsp;基本实现： Detectron 2[<sup>2</sup>](#refer-anchor-2)  
&emsp;图像增强[<sup>3</sup>](#refer-anchor-3)：基于opencv库的实现  
&emsp;图像标注：labelme  
&emsp;数据格式转换[<sup>4</sup>](#refer-anchor-4)：labelme -> coco, coco -> csv, csv -> coco

## 2. 代码使用说明
### &emsp;2.1 操作系统与环境准备
#### &emsp;&emsp;2.1.1 操作系统
&emsp;&emsp;&emsp;由于Detectron 2[<sup>2</sup>](#refer-anchor-2)环境的限制，这里使用Linux系统（个人使用的是OpenSUSE发行的Linux系统）

#### &emsp;&emsp;2.1.2  `python`环境  
&emsp;&emsp;&emsp;使用`Anaconda 3`发行版.下载地址：  
https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh  
下载完后在**下载的目录**打开终端并输入以下命令：
```shell
sh Anaconda3-2020.07-Linux-x86_64.sh
```
***注意：最后的提示是是否要添加`path`到环境变量，请输入`yes`以便后续的操作。***

### &emsp;2.2 数据准备
#### &emsp;&emsp;2.2.1 数据的标注
&emsp;&emsp;&emsp;采用`labelme`的python包进行标注。  
 1. 安装`labelme`包  
打开终端并输入以下命令：
```shell
pip install labelme
```
 2. 打开`labelme`  
打开终端并输入以下命令：
```shell
labelme
```
即可弹出窗口界面。

 3. `labelme` 的使用
   参见`Labelme 图像标记教程.pdf `

#### &emsp;&emsp;2.2.2 数据的转换
&emsp;&emsp;&emsp;将`labelme`标记的数据转换为`coco`数据集.  
 1. 将标记好的数据（**包括原图和生成的`json`文件**）放在主目录下的`pic`文件夹内
 2. 在主目录下打开终端，并输入命令
```shell
python labelme2coco.py
```
 3. 最终会在主目录下生成`coco`文件夹，里面已经包含了所有已标记的图片数据。


## 参考
<div id="refer-anchor-1"></div>

[1] [Mask R-CNN](https://arxiv.org/abs/1703.06870)

<div id="refer-anchor-2"></div>

[2] [Detectron 2](https://github.com/facebookresearch/detectron2)

<div id="refer-anchor-3"></div>

[3] [图像增强](https://github.com/niuxiaozhang/convert_dataset_to_coco)

<div id="refer-anchor-4"></div>

[4] [数据格式转换](https://github.com/niuxiaozhang/convert_dataset_to_coco)