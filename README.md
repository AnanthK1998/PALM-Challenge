# Detection of Pathological Myopia

This was a challenge in ISBI 2019 conference. This challenge has four subtasks : Image classification, Disc Segmentation, Lesion Segmentation and Fovea Localization. I have done the first three subtasks in this repository.

We have approached the classification subtask with the ResNet-50 and the DenseNet-121 architectures.

The segmentation subtasks were approached with the UNet architecture with ResNeXt-101 encoder using Keras.

Pathological or Degenerative Myopia is quite different from the simple refractive myopia or nearsightedness that affects so many people around the world. Pathological myopia is an extremely high amount of nearsightedness that causes a major alteration of the shape or globe of the eye, which may lead to profound vision loss. Pathological myopia causes the eye to elongate, which in turn stretches and thins the retina and the sclera of the eye. This leads to a bulging of the posterior portion of the eyeball.

![alt text](https://images.squarespace-cdn.com/content/v1/52c9aac2e4b0ef830a22e002/1391735930833-9IPX37G07IBZ295Y08NV/ke17ZwdGBToddI8pDm48kJ7IdKXPYAEhE9WR4yBGzqVZw-zPPgdn4jUwVcJE1ZvWhcwhEtWJXoshNdA9f1qD7Xj1nVWs2aaTtWBneO2WM-vBBtjjpcuJwmFyrVZD0BuVy_5rmmmAta3U8h3a_D1KxQ/myopic_mac_degen.jpg?format=500w)

This project focuses on the investigation and development of algorithms associated with the diagnosis of Pathological Myopia in fundus photos from PM patients. The goal of the project is to evaluate and compare Deep Learning algorithms for the detection of pathological myopia on a common dataset of retinal fundus images.


In this project, I have used two Deep Learning Architectures:-

1. Residual Neural Network(ResNet50)

2. DenseNet(DenseNet121)

## Residual Neural Network(ResNet)

A Deeper Convolutional Neural Network is able to extract complex features, but as we go deeper it becomes difficult to train because of  vanishing gradient problem — as the gradient is back-propagated to earlier layers, repeated multiplication may make the gradient infinitively small. As a result, as the network goes deeper, its performance gets saturated or even starts degrading rapidly.

The core idea of Residual Neural Network is introducing a so-called “identity shortcut connection” that skips one or more layers. These shortcut connection or skip connections are used to solve tne vanishing gradient problem.

![alt text](https://miro.medium.com/max/510/1*ByrVJspW-TefwlH7OLxNkg.png)

The formulation of F(x)+x can be realized by feedforward neural networks with shortcut connections. Shortcut connections are those skipping one or more layers shown in Figure. The shortcut connections perform identity mapping, and their outputs are added to the outputs of the stacked layers. By using the residual network, there are many problems which can be solved such as:

* ResNets are easy to optimize, but the “plain” networks (that simply stack layers) shows higher training error when the depth increases.
* ResNets can easily gain accuracy from greatly increased depth, producing results which are better than previous networks.



## DenseNet

DenseNet architecture is a logical extension of ResNet. While ResNet uses skip connections which every alternative layer i.e. you merge (additive) a previous layer into a future layer, DenseNet on the other hand connects each layer to every other layer in a feed-forward fashion i.e. concatenating outputs from the previous layers instead of using the summation.

Traditional convolutional networks with L layers have L connections — one between each layer and its subsequent layer — DenseNet has L(L+1)/ 2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers.

![alt text](https://miro.medium.com/max/593/1*GeK21UAbk4lEnNHhW_dgQA.png)
 
 As we can see from the Figure, how DensNet connects each layer to every other layer. DenseNets have several  advantages:
* They alleviate the vanishing-gradient problem.
* Strengthen feature propagation.
* Encourage feature reuse.
* Substantially reduce the number of parameters.







