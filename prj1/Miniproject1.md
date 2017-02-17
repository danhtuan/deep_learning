# Mini Project 1: MNIST data training using Neural Network
 - Tuan Nguyen
 - Deep Learning - Spring 2017
 - Advisor: Dr. Martin Hagan
 - Link: https://github.com/danhtuan/deep_learning/
 
## Abstract

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) provides a training set of 60,000 handwritten digits and a test set of 10,000 handwritten digits. The images have a size of 28×28 pixels. We want to train a Neural Network to recognize handwritten digits.

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
trainInputs = trainInputs:cuda()
trainTargets = trainTargets:cuda()
validInputs = validInputs:cuda()
validTargets = validTargets:cuda()
testInputs = testInputs:cuda()
testTargets = testTargets:cuda()
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

## 3. Mini-batches vs Stochastic Gradient Descent
Basically, we have 3 ways to feed data to NN:
* **Batch Gradient Descent (BGD)** : feed all data points in each iteration
* **Mini-batches Gradient Descent (MBGD)** : feed 1 < b < ALL data points in each iteration
* **Stochastic Gradient Descent (SGD)** : feed only 1 data point in each iteration

The original program is set up to perform SGD. In this section, we experiment the NN using MBGD.
* Using package 'optim'
* Divide the dataset into mini-batches to feed to NN
* Vary batch-size

Here is the implementation using Minibatches:
```lua
params, gradParams = module:getParameters()
local optimState = {learningRate = 0.1}
local batchSize = 100 -- batch size


function trainEpoch(module, criterion, inputs, targets)        
    local numBatch = inputs:size(1)/batchSize -- number of batches
    print(inputs[1]:size(1)..'x'..inputs[1]:size(2)..'x'..inputs[1]:size(3))
    for i = 1, numBatch do
      local idx = math.random(1, numBatch) -- random minibatch
      local batchInputs = torch.DoubleTensor(batchSize, 1, 28, 28)
      local batchLabels = torch.DoubleTensor(batchSize)
      --create mini-batch
      for j = 1, batchSize do
        local ref_idx = (idx - 1)  * batchSize + j
        batchInputs[j] = inputs[ref_idx]
        batchLabels[j] = targets:narrow(1, ref_idx, 1)          
      end
      --train using mini-batch
      function feval(params)
        gradParams:zero()

        local outputs = module:forward(batchInputs)
        local loss = criterion:forward(outputs, batchLabels)
        local dloss_doutputs = criterion:backward(outputs, batchLabels)
        module:backward(batchInputs, dloss_doutputs)
        return loss, gradParams
      end
      optim.sgd(feval, params, optimState)
    end
end

```
Here is the output:

```
New maxima : 0.902500 @ 1.000000
Test Accuracy : 0.910500 
Duration: 78874.533891678ms
Program completed in 83.10 seconds (pid: 29171).
```
__NOTE__
* Mini-batches Gradient Descent converges faster than Stochastic Gradient Descent
* Have the mistmatch size problem (took me hours), temporarily fixed by setting number of epoch = 1. 

## 4. Number of layers vs. Number of Neurons

Current code to create a two-layer network:

```lua
-- Create a two-layer network
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax()) 
```

The number of weights and biases:
* Input Layer:

> 20 * 28 * 28 = 15,680 (weights) and 20 (biases)

* Output Layer:

> 10 * 20 = 200 (weights) and 10 (biases)

I experimented with 3 network configuration as following:
* 1 layers MLP
* 2 layers MLP
* 3 layers MLP
```lua
if opt.model == 'mlp2' then
        --1st layer
        module:add(nn.Linear(1*28*28, 20))
        module:add(nn.Tanh())
        --2nd layer
        module:add(nn.Linear(20, 10))
elseif opt.model == 'mlp3' then
        module:add(nn.Linear(1*28*28, 15))
        module:add(nn.Tanh())
        module:add(nn.Linear(15,10))
        module:add(nn.Tanh())
        --output layer
        module:add(nn.Linear(10,10))
elseif opt.model == 'linear' then
        module:add(nn.Linear(1*28*28, 10))
end

```

## 5. Gradient vs. Alternative functions


## 6. Conclusion
* Learned how to run the program on CPU and GPU, can compare the performance
* Learned the difference between SGD vs. Mini-batches
* Learned and compared number of layers vs number of neurons and saw __deep__ >> __wide__ (in this case)
* Learned another function besides of gradient descent
