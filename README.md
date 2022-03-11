# Deformable-CNN
> 官方（torchvision）源码：
>
> ​		 	https://pytorch.org/vision/stable/_modules/torchvision/ops/deform_conv.html

一种可变形卷积网络模块，与Conv2D一样，但是多了一半参数。常用于检测形状易变的物体，在目标检测当中使用的很多。

![Deformable Conv ](https://gitee.com/lpengsu/pic-go/raw/master/img/Deformable%20Conv%20%E5%8F%82%E6%95%B0%E5%9B%BE.jpeg)

## 在MNIST数据集中使用

1. 使用pytorch框架完善lighting Data Module

2. 分别使用常规2D 卷积和 可变形卷积 完善模型

3. 对模型训练相同的次数

   ![image20220310105355599](https://gitee.com/lpengsu/pic-go/raw/master/img/image-20220310105355599.png)

   通过对比可以发现，可变形卷积（蓝色）在简单的模型中**较优于**普通卷积。

4. 对数据采用旋转或翻转的变换作为测试集

5. 对比两种模型对新的数据的识别程度

![image20220310105648791](https://gitee.com/lpengsu/pic-go/raw/master/img/image-20220310105648791.png)

 在对测试数据进行旋转和翻转进行测试之后，通过对比可以发现仍然是可变形卷积的精度要优于普通卷积，说明可变形卷积有一定的抗干扰能力。
