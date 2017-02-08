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

-- first layer weight
model:add(nn.Linear(R, S1));

--performance index
local crit = nn.MSECriterion();

local Q = 3; --#data points

local P = torch.Tensor(Q, R)
P[{{}, {1}}] = torch.Tensor({-1, 0, 1})

T = torch.Tensor({-1.5, 0.5, 2.5})
--tranining plot data


local runtest = function()
  local numIteration = 1000
  local  sse = torch.Tensor(2, numIteration)
  local params, gradParams = model:getParameters()
  local optimState = {learningRate = 0.01}
  require 'optim'
  for epoch = 1, 1000 do
    sse[1][epoch] = epoch
    --local dloss_dout = 10000
    local function feval(params)
      gradParams:zero()
      local A = model:forward(P)
      local loss = crit:forward(A, T)
      sse[2][epoch] = loss
      dloss_dout = crit:backward(A, T)
      model:backward(P, dloss_dout)
      return loss, gradParams
    end
    
    if(epoch == 1 or torch.norm(gradParams) > 0.01) then
      optim.sgd(feval, params, optimState)
    else
      break
    end        
  end
  --Plot error vs. iteration
  require 'gnuplot'
  gnuplot.plot(sse[{1, {}}], sse[{2, {}}])
end

--CPU Test
--require 'sys'
--tick = sys.clock()
runtest()

function test()
  local ND = 31
  local Pt = torch.Tensor(ND, R)
  Pt[{{}, {1}}] = torch.range(-1.5, 1.5, 0.1)
end
--tock = sys.clock()--
--duration = tock - tick
--print('CPU TIme: '..(duration * 1000)..'ms)

