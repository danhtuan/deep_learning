--Neural Network
--Lab 1 - Problem 5
--@author @danhtuan
--@date Feb 7, 17
--@desc

require 'nn'
-- multi-layer
local model = nn.Sequential();
-- parameters
local R = 1;-- #inputs
local S1 = 1; -- #neurons in 1st layer
local S2 = 20;-- #neurons in 2nd layer

-- first layer weight
model:add(nn.Linear(R, S1));
--transfer function f1
model:add(nn.Tanh());

--second layer W2
model:add(nn.Linear(S1, S2));

--performance index
local crit = nn.MSECriterion();

local Q = 21;

local P = torch.Tensor(Q, R);

P[{}, {1}] = torch.range(-2, 2, 0.2);

T = torch.sin(P)

local runtest = function()
  local params, gradParams = model:getParameters()
  local optimState = {learningRate = 0.01}
  require 'optim'
  for epoch = 1, 10000 do
    local function feval(params)
      gradParams:zero()
      local A = model:forward(P)
      local loss = crit:forward(A, T)
      local dloss_dout = crit:backward(A, T)
      model:backward(P, dloss_dout)
      return loss, gradParams
    end
    optim.sgd(feval, params, optimState)
  end
end

--CPU Test
require 'sys'
tick = sys.clock()
runtest()
tock = sys.clock()
duration = tock - tick
print('CPU TIme: '..(duration * 1000)..'ms)

