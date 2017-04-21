import numpy as np
import theano
import theano.tensor as T
rng = np.random

input = np.array([[1],[2],[3],[4]])
target = np.array([[3],[5],[7],[9]])

p = T.dmatrix("p")
t = T.dmatrix("t")

# initialize the weight
w = theano.shared(rng.randn(1), name="w")

# initialize the bias
b = theano.shared(0., name="b")

iterations = 1000

a = w*p + b

e = t - a

e2 = T.sqr(e)

perf = T.sum(e2)

gw, gb = T.grad(perf, [w, b])

train = theano.function(
          inputs=[p,t],
          outputs=[a, perf],
          updates=((w, w - 0.01 * gw), (b, b - 0.01 * gb)))
predict = theano.function(inputs=[p], outputs=a)
perform = theano.function(inputs=[p,t], outputs=perf)

# Train,
for i in range(iterations):
    pred, err = train(input, target)

print("Final model:")
print(w.get_value())
print(b.get_value())
