# Mini Project 1: MNIST data training using Neural Network
 - Tuan Nguyen
 - Deep Learning - Spring 2017
 - Advisor: Dr. Martin Hagan
 - Link: https://github.com/danhtuan/deep_learning/
 
## 1. Run the program on CPU
Here is the screenshot of the Output:

![CPU Screen](imgs/cpu_screen.png)

__NOTE__
- It took **881.89 seconds** to finish
- The accuracy on test set is lower than on validation set, which is reasonable because test set is *NOT* used to train

## 2. Modify the code to run on GPU
To make it run on GPU, following code added to the original code:
```lua
require 'cunn'
module:cuda()
criterion:cuda()
trainInputs:cuda()
trainTargets:cuda()
validInputs:cuda()
validTargets:cuda()
testInputs:cuda()
testTargets:cuda()
```

__NOTE__ When running the new code, following error message can appear:

>cannot convert 'struct THLongTensor *' to 'struct THCudaLongTensor *'

The reason for this is due to **nn, cunn, torch, cutorch** are out-of-date. The bug has been found and fixed in the new update. Please update using `luarocks` as following:

```
luarocks install torch
luarocks install nn
luarocks install cutorch
luarocks install cunn
```

After fixing, here is the screenshot of the output:

![GPU_Screen](imgs/gpu_screen.png)

__NOTE__ 
* It took **734.82 seconds** to finish, *which is a little bit faster than CPU version*

## Minimatches vs Stochastic Gradient Descent

