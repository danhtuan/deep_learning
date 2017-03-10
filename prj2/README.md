# Miniproject 2: Caffe and MNIST dataset
* Tuan Nguyen
* Spring 17
* Deep learning - Dr. Martin Hagan

## 1. Setting up
* Download 4 data files and unzip
* Modify create_mnist.sh
* Modify prototxt files
| <img src="figure_1.png" width=300/>|<img src="figure_2.png" width=300/>|
| <img src="figure_3.png" width=300/>|<img src="figure_4.png" width=300/>|
| <img src="figure_5.png" width=300/>||
## 2. Investigate the kernels

Two convolution layers are defined in file `lenet_train_test.prototxt` as following:

```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

## 3. Performance CVN vs. Multilayer Networks
### Accuracy
* Convolution Networks

* Multiplayers Networks
