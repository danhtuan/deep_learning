# Lab 3 Report
 * Tuan Nguyen
 * Deep Learning Spring 2017
 * Dr. Martin Hagan
 
## A.Questions
### 1. What are ”placeholders”, and how are they used? Give some examples.

A `placeholder` is simply a variable that we will assign data to at a later date. It allows us to create our operations and build our computation graph, without needing the data. In TensorFlow terminology, we then feed data into the graph through these placeholders. Examples:

```python
import tensorflow as tf
  
 a = tf.placeholder("float")
 b = tf.placeholder("float")
  
 y = tf.mul(a, b)
  
 sess = tf.Session()
  
 print sess.run(y, feed_dict={a: 3, b: 3})
```
In this example, we declared 2 placeholders a and b and then multiply them together (without needing the data) with `tf.mul`. When data is available, we `feed` them to placeholders and run the session. Another example from TensorFlow tutorial shows why feeding is needed:

```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

### 2. What are ”variables”, and how are they used?

`Variables` are in-memory buffers containing tensors. We use variables to hold and update parameters (weights and bias, in specific). Here is an example from TensorFlow tutorial:

```python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
```

### 3. What are ”tensors” in TensorFlow?

TensorFlow programs use a tensor data structure to represent all data -- only tensors are passed between operations in the computation graph. Tensor can be thought as an n-dimensional array or list. A tensor has a static type, a rank, and a shape.

### 4. Explain how TensorFlow uses a dataflow graph to represent networks and operations. 
We might think of TensorFlow Core programs as consisting of two discrete sections:
 * Building the computational graph.
 * Running the computational graph.

TensorFlow is constructed around the basic idea of building and manipulating a computational graph, representing symbolically the numerical operations to be performed. TensorFlow can be seen as a library for numerical computation using data flow graphs. Think of a computational graph as a network of nodes, with each node known as an operation, running some function that can be as simple as addition or subtraction to as complex as some multi variate equation.

Here is an example that create a very simple graph that includes just three nodes:

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)
```

### 5. What are the nodes of the graph? What are the edges?
The nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors), which interconnect the nodes.

### 6. How do you run the graph?
To run the graph, we can use `tf.Session()` as following example:

```python
sess = tf.Session()
print(sess.run(node3))
```

## B.Code
### 1. Explain one_layer.py
```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```
--> Create a linear model with 2 parameters W, b and 1 variable x
```python
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
```
--> Run the model with init values
```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```
--> Run the model, compare with target y and compute the lost then assign W, b with correct value
```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
```
--> Reset model, then train the model using Gradient Descent Optimizer with 1000 epoches
```
y([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
```
--> The trained model have W and b are very close to the correct one (-1, 1) above

### 2. Modify to solve HW3
### 3. Explain count.py
```python
train_input = ['{0:015b}'.format(i) for i in range(2 ** 15)]
shuffle(train_input)
train_input = [map(int, i) for i in train_input]
ti = []
for i in train_input:
    temp_list = []
    for j in i:
        temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * 16)
    temp_list[count] = 1
    train_output.append(temp_list)

NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]  # everything beyond 10,000

train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]  # till 10,000
```
--> This code create a dataset of 2^15 points then divide it into two sets:
 * Training dataset: 10000 points
 * Test set: the rest

```
Epoch 500 error 0.2%
[[  2.08623234e-08   5.41389156e-10   3.66248004e-11   4.04669942e-09
    1.67097426e-07   7.02460952e-08   4.77521489e-08   6.69241535e-06
    3.15301027e-03   9.96461332e-01   3.78547993e-04   3.23926024e-08
    1.59584594e-08   1.02191908e-10   1.67648491e-08   9.09400022e-09]]
[[  4.47081157e-11   8.38611038e-12   1.21016919e-10   1.33419387e-09
    1.12379425e-07   5.13746694e-04   9.99075174e-01   4.07108018e-04
    6.65679345e-10   7.18384033e-11   3.69486656e-06   3.31567218e-10
    9.76393767e-15   2.08210730e-16   2.20294334e-12   5.55894246e-08]]
```
