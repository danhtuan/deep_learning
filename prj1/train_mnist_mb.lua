require 'dp'
require 'sys'
require 'dpnn'
require 'optim'

-- Load the mnist data set
ds = dp.Mnist()

-- Extract training, validation and test sets
trainInputs = ds:get('train', 'inputs', 'bchw')
trainTargets = ds:get('train', 'targets', 'b')
validInputs = ds:get('valid', 'inputs', 'bchw')
validTargets = ds:get('valid', 'targets', 'b')
testInputs = ds:get('test', 'inputs', 'bchw')
testTargets = ds:get('test', 'targets', 'b')

opt = {
	batchSize = 100,
	learningrate = 0.1,
	momentum = 0.9,
	model = 'linear', 
	optimization = 'SGD',
	maxIter = 100,
	num_epoch = 2 
}
--use floats, for SGD
if opt.optimization == 'SGD' then
	torch.setdefaulttensortype('torch.FloatTensor')
end
-- Create a two-layer network
module = nn.Sequential()
module:add(nn.Convert('bchw', 'bf')) -- collapse 3D to 1D

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
module:add(nn.LogSoftMax()) 
params, gradParams = module:getParameters()

-- Use the cross-entropy performance index
criterion = nn.ClassNLLCriterion()

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

function trainEpoch(module, criterion, inputs, targets)
    for t = 1, inputs:size(1), opt.batchSize do
	--create minibatch
	local inputBatch = torch.Tensor(opt.batchSize, 1, 28, 28)
	local targetBatch = torch.Tensor(opt.batchSize)
	local k = 1
	for i = t, math.min(t + opt.batchSize - 1, inputs:size(1)) do
	    --load new sample
	    local input = inputs[i] 
	    local target = targets[i]
	    inputBatch[k] = input
	    targetBatch[k] = target
	    k = k + 1
	end
	--create feval
	local feval = function(params)
	  gradParams:zero()
	  local outputs = module:forward(inputBatch)
	  local loss = criterion:forward(outputs, targetBatch)
	  local dloss_doutputs = criterion:backward(outputs, targetBatch)
	  module:backward(inputBatch, dloss_doutputs)
	  return loss, gradParams
	end

	--
	if opt.optimization == 'LBFGS' then
		lbfgsState = lbfgsState or {
			maxIter = opt.maxIter,
			lineSearch = optim.lswolfe
		}
		optim.lbfgs(feval, params, lbfgsState)
		--disp report
		print('LBFGS step')
		print(' - nb of function eval:'..lbfgsState.funcEval)
	elseif opt.optimization == 'SGD' then
		optimState = optimState or {
			learningRate = opt.learningrate,
			momentum = opt.momentum,
			learningRateDecay = 5e-7
		}
		optim.sgd(feval, params, optimState)
	end
    end
end

--Run the training
tick = sys.clock()

bestAccuracy, bestEpoch = 0, 0
wait = 0
for epoch=1, opt.num_epoch do
   trainEpoch(module, criterion, trainInputs, trainTargets)
   local validAccuracy = classEval(module, validInputs, validTargets)
   if validAccuracy > bestAccuracy then
      bestAccuracy, bestEpoch = validAccuracy, epoch
      --torch.save("/path/to/saved/model.t7", module)
      print(string.format("New maxima : %f @ %f", bestAccuracy, bestEpoch))
      wait = 0
   else
      wait = wait + 1
      if wait > opt.num_epoch then break end
   end
end
testAccuracy = classEval(module, testInputs, testTargets)
print(string.format("Test Accuracy : %f ", testAccuracy))

tock = sys.clock()
duration = tock -tick
print('Duration: '..(duration * 1000)..'ms')
