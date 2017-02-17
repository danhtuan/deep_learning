require 'dp'

-- Load the mnist data set
ds = dp.Mnist()

-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw')
trainTargets = ds:get('train', 'targets', 'b')
--print(trainTargets)
validInputs = ds:get('valid', 'inputs', 'bchw')
validTargets = ds:get('valid', 'targets', 'b')
testInputs = ds:get('test', 'inputs', 'bchw')
testTargets = ds:get('test', 'targets', 'b')

-- Create a two-layer network
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D
module:add(nn.Linear(1*28*28, 20))
module:add(nn.Tanh())
module:add(nn.Linear(20, 10))
module:add(nn.LogSoftMax()) 

-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion()



require 'optim'
-- allocate a confusion matrix
cm = optim.ConfusionMatrix(10)
-- create a function to compute 
function classEval(module, inputs, targets)
   cm:zero()
   for idx=1,inputs:size(1) do
      local input, target = inputs[idx], targets[idx]
      local output = module:forward(input)
      cm:add(output, target)
   end
   cm:updateValids()
   return cm.totalValid
end

 require 'dpnn'
 
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

--Run the training
require 'sys'
tick = sys.clock()

bestAccuracy, bestEpoch = 0, 0
wait = 0
for epoch=1,1 do
   trainEpoch(module, criterion, trainInputs, trainTargets)
   local validAccuracy = classEval(module, validInputs, validTargets)
   if validAccuracy > bestAccuracy then
      bestAccuracy, bestEpoch = validAccuracy, epoch
      --torch.save("/path/to/saved/model.t7", module)
      print(string.format("New maxima : %f @ %f", bestAccuracy, bestEpoch))
      wait = 0
   else
      wait = wait + 1
      if wait > 1 then break end
   end
end
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))

tock = sys.clock()
duration = tock -tick
print('Duration: '..(duration * 1000)..'ms')
