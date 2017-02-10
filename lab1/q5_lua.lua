--Question 5
--@author @danhtuan
--@date Feb 7, 17
require 'gnuplot'
require 'torch'
--gnuplot.closeall()
--Init
local R = 1 --#inputs
local Q = 3 --#data points
local lr = 0.1 --learning rate
--w & b
local x = torch.Tensor(R + 1, 1):fill(0) --homogenious 
--input
local P = torch.Tensor(Q, 1)
P[{{}, {1}}] = torch.Tensor({-1, 0, 1})
--targets
local T = torch.Tensor(Q, 1)
T[{{}, {1}}] = torch.Tensor({-1.5, 0, 2.5})

local G = torch.Tensor({{-1, 0, 1}, {1, 1, 1}}):t()
--A, d, c
local c = torch.mm(T:t(), T)
local d = torch.mm(G:t(), T):mul(-2)
local A = torch.mm(G:t(), G):mul(2)
--F(x) the loss function (criterion)
local function floss()
  local F = x:t() * A * x
  F = F + torch.mm(d:t(), x)
  F = F + c
  return F
end

--gradient
local function df(var)
  local delta = torch.mm(A, x)
  delta = delta + d
  return delta
end

--update w & b
local function update(grad)
  x = x - grad:mul(lr)
end
--train the network
local runtest = function()
  local sse = torch.Tensor(10)
  --compute gradient
  local grad =  df(x)
  local iter = 1
  sse[iter] = floss()
  while(torch.norm(grad) >= 0.01) do
    iter = iter + 1
    --update network
    update(grad)
    sse[iter] = floss()
  end
  gnuplot.figure()
  gnuplot.title('SSE vs. Iteration')
  gnuplot.xlabel('iterations')
  gnuplot.ylabel('SSE')
  gnuplot.plot(sse)
end
runtest()
--test the network
local AP = torch.range(-1.5, 1.5, 0.1)
local AT = AP:mul(x[1][1]):add(x[2][1])
gnuplot.figure()
gnuplot.title('Network responses vs. targets')
gnuplot.plot({AP, AT, '-'}, {P:select(2,1), T:select(2,1), '+'})
