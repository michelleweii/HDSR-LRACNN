# HDSR-LRACNN

## Abstract
In this work, a model based on convolutional neural networks (CNNs) is proposed to solve handwritten digit string recognition (HDSR) problem in one pipeline. We design a feature extraction network using up-sampling and highway mechanism, which can effectively handle the low resolution images. We also introduce digital proposal network with reference to anchor boxes of digital shape. This model is end-to-end and can recognize skewed, connecting, touching and arbitrary length digit strings. And besides, we propose two effective tricks in the process of non-maximum suppression (NMS) to improve accuracy greatly. In the experiments, we collected a large number of samples from the tablet with the help of college students and employees. These samples include different length digit strings with class and bounding box labels. We also use the synthetic strings derived from MNIST for training. Experimental results show that arbitrary length strings can be recognized with 94.75% accuracy and 99.21% mAP, and the highest accuracy of 99.3% is obtained in the fixed-length case. Furthermore, the recognition speed of our model is about 6 times faster than Faster R-CNN’s, towards real-time.

# Model Overview

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/model_overview.png" width="700" alt="model_overview">

## Modified CNN (LRAN)

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/structure.png" width="719" alt="structure">

# Heuristic NMS

实验过程中，分析了bad cases，主要原因在于字符级别的检测和传统目标检测不一样，传统目标检测中物体会分的比较开，但是字符都是挨在一起的。NMS是对同类别物体做抑制的，先选中一个物体，比如是“猫”。(>^ω^<)喵，对所有类别为“猫”的bbox按照score降序排列，选中最大的bbox，用剩余的bbox和它做对比，IOU==0.5作为阈值，大于0.5的说明两个bbox重合度较高，那么删除掉第二个bbox，依次类推，对“猫”进行bbox的nms后，再对“狗”的bbox进行nms。

所以上述流程存在一个问题，在字符级别的检测中，假如对“1”进行检测产生了很多候选的bbox，那么nms的时候两个框尽管离得很近但是iou的阈值并没有达到，所以会残留很多的bbox，如果“1”后面接的是“0”，“0”的候选框比较偏正方形，那么是很有可能和前面“1”产生冗余的bbox的iou>0.5的。

这个采用一个启发式的思路：我们改变nms对不同类别分别做nms的策略，将“1”，“2”，...，“9”这10个类别当做是“一类“物体，一起进行nms，字符与字符之间相互抑制。采用这个启发式的nms，实验结果在定长的情况下，accuracy上升了1.25%to7%。 

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/NMS.png" width="450" alt="structure">

# Datasets

## HDS5
<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/HDS5.png" width="400" alt="HDS5">

## ICFHR 2014
<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/HDSRC2014.png" width="400" alt="ICFHR 2014">

# Experiment Results
在HDS5上的检测结果

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/HDS5实验结果示意图.PNG" width="400" alt="results">
在ICFHR 2014上的检测结果

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/Results.png" width="400" alt="results">

优势在于可以检测这种不是一条水平线的数据，ctpn这种行检测就无法完成。

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/430040-0396-15.png" width="100" alt="results1">
<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/7268671-0948-22.png" width="150" alt="results2">



# Accept Notication

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/i3d.png" width="700" alt="accept">
