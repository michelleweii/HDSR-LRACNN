# HDSR-LRACNN
In this work, a model based on convolutional neural networks (CNNs) is proposed to solve handwritten digit string recognition (HDSR) problem in one pipeline. We design a feature extraction network using up-sampling and highway mechanism, which can effectively handle the low resolution images. We also introduce digital proposal network with reference to anchor boxes of digital shape. This model is end-to-end and can recognize skewed, connecting, touching and arbitrary length digit strings. And besides, we propose two effective tricks in the process of non-maximum suppression (NMS) to improve accuracy greatly. In the experiments, we collected a large number of samples from the tablet with the help of college students and employees. These samples include different length digit strings with class and bounding box labels. We also use the synthetic strings derived from MNIST for training. Experimental results show that arbitrary length strings can be recognized with 94.75% accuracy and 99.21% mAP, and the highest accuracy of 99.3% is obtained in the fixed-length case. Furthermore, the recognition speed of our model is about 6 times faster than Faster R-CNN’s, towards real-time.

# Datasets--HDS5

# Model Overview

# Experiment Result


# Accept Notication
<img src="https://github.com/michelleweii/HDSR-LRACNN/blob/master/pic/accept.jpg" width="900" alt="录用通知">
