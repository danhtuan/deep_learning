### Final Project Report
# Training Convolutional Neural Network using different floating point formats
  * Tuan Nguyen
  * Deep Learning Spring 2017
  * Dr. Martin Hagan
  
## 1. Introduction

In recent years, Deep Learning or Deep Neural Network (DNN) has grown tremendously in its popularity and usefulness []. However, DNN is computationally intensive, power-hungry and very often limited by hardware capability. For many years, single-precision (float32) and double-precision (float64) floating point formats have been widely used as the default formats for DNN. 

However, using lower precision is growing rapidly as a trend in DNN research. Recent research [][][] shows that low-precision and very low precision are sufficient for training and running DNN. In addition, in 2016, NVIDIA introduced Pascal GPU architecture and CUDA 8 SDK that fully support half-precision (float16) floating point format and mixed-precision computing. The lower memory, higher speed, and less power consumption are main motivations behind this trending. 

This project is a small research to examine how different floating-point precisions affect on DNN's accuracy, speed and memory. Due to the limited time, a simplified version of Convolutional Neural Network based on LeNet5[] has been selected as the network to train. The selected datasets are MNIST[] and CIFAR-10[], which are well-known benmarking datasets for machine learning algorithms.

The rest of report is oraganized as following: Section 2 is a very short introduction to Floating Point Formats used in this research. Section 3 is a description about MNIST and CIFAR-10 datasets and Section 4 describes about Convolutional Neural Network design. Section 5 shortly describes about the hardware/software configuration in the experiment and results will be shown in Section 6. The final section is a short conclusion.
## 2. Floating Point Formats


## 3. Datasets

### 3.1 MNIST Dataset
### 3.2 CIFAR-10

The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

Here are the classes in the dataset, as well as 10 random images from each:

<img src="cifar10.png" alt="cifar10" style="width: 100px;"/>			

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

## 4. Convolutional Neural Network

## 5. Experimental Setup


## 6. Experimental Results

## 7. Conclusion

## Reference
