# Lab 3 Report
 * Tuan Nguyen
 * Deep Learning Spring 2017
 * Dr. Martin Hagan
 
## Questions
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


### 4. Explain how TensorFlow uses a dataflow graph to represent networks and operations. 

### 5. What are the nodes of the graph? What are the edges?

### 6. How do you run the graph?

## Code
