# HDSR-LRACNN

## Abstract

In this work, a model based on convolutional neural networks (CNNs) is proposed to solve handwritten digit string recognition (HDSR) problem in one pipeline. We design a feature extraction network using up-sampling and highway mechanism, which can effectively handle the low resolution images. We also introduce digital proposal network with reference to anchor boxes of digital shape. This model is end-to-end and can recognize skewed, connecting, touching and arbitrary length digit strings. And besides, we propose two effective tricks in the process of non-maximum suppression (NMS) to improve accuracy greatly. In the experiments, we collected a large number of samples from the tablet with the help of college students and employees. These samples include different length digit strings with class and bounding box labels. We also use the synthetic strings derived from MNIST for training. Experimental results show that arbitrary length strings can be recognized with 94.75% accuracy and 99.21% mAP, and the highest accuracy of 99.3% is obtained in the fixed-length case. Furthermore, the recognition speed of our model is about 6 times faster than Faster R-CNN’s, towards real-time.

# Model Overview

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/model_overview.png" width="700" alt="model_overview">

## Modified CNN

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/structure.png" width="700" alt="structure">

# Datasets

## HDS5

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/HDS5.png" width="400" alt="HDS5">

## ICFHR 2014

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/HDSRC2014.png" width="400" alt="ICFHR 2014">

# Experiment Results

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/Results.png" width="400" alt="results">

## ICFHR 2014中的small test case

### case1

优势在于可以检测这种不是一条水平线的数据，ctpn这种行检测就无法完成。

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/430040-0396-15.png" width="100" alt="results1">

### case2

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/7268671-0948-22.png" width="150" alt="results2">


# NMS

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/NMS.png" width="400" alt="structure">

# Accept Notication

<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/i3d.png" width="600" alt="accept">
