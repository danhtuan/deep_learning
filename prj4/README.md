# MiniProrject 4 Report
  * Tuan Nguyen
  * Deep Learning Spring 2017
  * Dr. Martin Hagan

## Run the program DBN.py and investigate and verify its performance
The program seemed to run forever with default epoch parameters. Looked at DBN tutorial, it said that 
```
On an Intel(R) Xeon(R) CPU X5560 running at 2.80GHz, using a multi-threaded MKL library 
(running on 4 cores), pretraining took 615 minutes with an average of 2.05 mins/(layer * epoch). 
Fine-tuning took only 101 minutes or approximately 2.20 mins/epoch.
```
To observe how it works, I have changed the number of pre-training epoch from 100-->10 and number of training epoch from 1000-->100. Here is the output from console: 

```
... loading data
... building the model
... getting the pretraining functions
... pre-training the model
Pre-training layer 0, epoch 0, cost  -98.5364928294
Pre-training layer 0, epoch 1, cost  -83.8445576167
Pre-training layer 0, epoch 2, cost  -80.6960564003
Pre-training layer 0, epoch 3, cost  -79.0383977598
Pre-training layer 0, epoch 4, cost  -77.9289304769
Pre-training layer 0, epoch 5, cost  -77.0886402508
Pre-training layer 0, epoch 6, cost  -76.4052954432
Pre-training layer 0, epoch 7, cost  -75.8301243604
Pre-training layer 0, epoch 8, cost  -75.3491888953
Pre-training layer 0, epoch 9, cost  -74.9350097361
Pre-training layer 1, epoch 0, cost  -259.404030997
Pre-training layer 1, epoch 1, cost  -235.102177549
Pre-training layer 1, epoch 2, cost  -230.004528048
Pre-training layer 1, epoch 3, cost  -227.267256281
Pre-training layer 1, epoch 4, cost  -225.547066512
Pre-training layer 1, epoch 5, cost  -224.261841755
Pre-training layer 1, epoch 6, cost  -223.306645882
Pre-training layer 1, epoch 7, cost  -222.558303674
Pre-training layer 1, epoch 8, cost  -221.948449722
Pre-training layer 1, epoch 9, cost  -221.412730343
Pre-training layer 2, epoch 0, cost  -75.2394828392
Pre-training layer 2, epoch 1, cost  -63.7686715783
Pre-training layer 2, epoch 2, cost  -61.6299658272
Pre-training layer 2, epoch 3, cost  -60.5279521274
Pre-training layer 2, epoch 4, cost  -59.8817089664
Pre-training layer 2, epoch 5, cost  -59.4089929983
Pre-training layer 2, epoch 6, cost  -59.0289154057
Pre-training layer 2, epoch 7, cost  -58.7553293121
Pre-training layer 2, epoch 8, cost  -58.5486016625
Pre-training layer 2, epoch 9, cost  -58.3210807635
The pretraining code for file DBN.py ran for 27.02m
... getting the finetuning functions
... finetuning the model
epoch 1, minibatch 5000/5000, validation error 3.740000 %
     epoch 1, minibatch 5000/5000, test error of best model 4.440000 %
epoch 2, minibatch 5000/5000, validation error 3.000000 %
     epoch 2, minibatch 5000/5000, test error of best model 3.310000 %
epoch 3, minibatch 5000/5000, validation error 2.670000 %
     epoch 3, minibatch 5000/5000, test error of best model 2.820000 %
epoch 4, minibatch 5000/5000, validation error 2.460000 %
     epoch 4, minibatch 5000/5000, test error of best model 2.480000 %
epoch 5, minibatch 5000/5000, validation error 2.390000 %
     epoch 5, minibatch 5000/5000, test error of best model 2.320000 %
epoch 6, minibatch 5000/5000, validation error 2.280000 %
     epoch 6, minibatch 5000/5000, test error of best model 2.180000 %
epoch 7, minibatch 5000/5000, validation error 2.170000 %
     epoch 7, minibatch 5000/5000, test error of best model 2.090000 %
epoch 8, minibatch 5000/5000, validation error 2.190000 %
epoch 9, minibatch 5000/5000, validation error 2.140000 %
     epoch 9, minibatch 5000/5000, test error of best model 1.900000 %
epoch 10, minibatch 5000/5000, validation error 2.120000 %
     epoch 10, minibatch 5000/5000, test error of best model 1.810000 %
epoch 11, minibatch 5000/5000, validation error 2.080000 %
     epoch 11, minibatch 5000/5000, test error of best model 1.800000 %
epoch 12, minibatch 5000/5000, validation error 2.030000 %
     epoch 12, minibatch 5000/5000, test error of best model 1.760000 %
epoch 13, minibatch 5000/5000, validation error 1.970000 %
     epoch 13, minibatch 5000/5000, test error of best model 1.720000 %
epoch 14, minibatch 5000/5000, validation error 1.920000 %
     epoch 14, minibatch 5000/5000, test error of best model 1.720000 %
epoch 15, minibatch 5000/5000, validation error 1.900000 %
     epoch 15, minibatch 5000/5000, test error of best model 1.680000 %
epoch 16, minibatch 5000/5000, validation error 1.840000 %
     epoch 16, minibatch 5000/5000, test error of best model 1.670000 %
epoch 17, minibatch 5000/5000, validation error 1.770000 %
     epoch 17, minibatch 5000/5000, test error of best model 1.630000 %
epoch 18, minibatch 5000/5000, validation error 1.770000 %
     epoch 18, minibatch 5000/5000, test error of best model 1.610000 %
epoch 19, minibatch 5000/5000, validation error 1.750000 %
     epoch 19, minibatch 5000/5000, test error of best model 1.590000 %
epoch 20, minibatch 5000/5000, validation error 1.720000 %
     epoch 20, minibatch 5000/5000, test error of best model 1.600000 %
epoch 21, minibatch 5000/5000, validation error 1.730000 %
epoch 22, minibatch 5000/5000, validation error 1.740000 %
epoch 23, minibatch 5000/5000, validation error 1.740000 %
epoch 24, minibatch 5000/5000, validation error 1.770000 %
epoch 25, minibatch 5000/5000, validation error 1.770000 %
epoch 26, minibatch 5000/5000, validation error 1.790000 %
epoch 27, minibatch 5000/5000, validation error 1.800000 %
epoch 28, minibatch 5000/5000, validation error 1.810000 %
epoch 29, minibatch 5000/5000, validation error 1.800000 %
epoch 30, minibatch 5000/5000, validation error 1.780000 %
epoch 31, minibatch 5000/5000, validation error 1.780000 %
epoch 32, minibatch 5000/5000, validation error 1.780000 %
epoch 33, minibatch 5000/5000, validation error 1.770000 %
epoch 34, minibatch 5000/5000, validation error 1.760000 %
epoch 35, minibatch 5000/5000, validation error 1.760000 %
epoch 36, minibatch 5000/5000, validation error 1.770000 %
epoch 37, minibatch 5000/5000, validation error 1.780000 %
epoch 38, minibatch 5000/5000, validation error 1.760000 %
epoch 39, minibatch 5000/5000, validation error 1.760000 %
Optimization complete with best validation score of 1.720000 %, 
obtained at iteration 100000, with test performance 1.600000 %
The fine tuning code for file DBN.py ran for 54.84m

IPython CPU timings (estimated):
  User   :    4046.45 s.
  System :     895.24 s.
Wall time:    4944.26 s.
```

It took 27 minutes to finish the pre-training phases and around 55 minutes to train the network. The test performance is 1.6% and the best validation score is 1.72% (error rate). 
## How is performance being measured in this network
Besides of the time, in the code, it used the validation/test loss, which are the mean of the loss taken all over the validation set and test set.

```python
393                 validation_losses = validate_model()
394                 this_validation_loss = numpy.mean(validation_losses, dtype='float64')
...
16                  test_losses = test_model()
417                 test_score = numpy.mean(test_losses, dtype='float64')
```

The single loss is the log likelihood of the logistic regression (output) layer.
```python
134         # compute the cost for second phase of training, defined as the
135         # negative log likelihood of the logistic regression (output) layer
136         self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
137 
138         # compute the gradients with respect to the model parameters
139         # symbolic variable that points to the number of errors made on the
140         # minibatch given by self.x and self.y
141         self.errors = self.logLayer.errors(self.y)
```

## Number of Layers/Neurons
The code to construct the DBN:

```python
318     dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
319               hidden_layers_sizes=[1000, 1000, 1000],
320               n_outs=10)
```
There are 3 RBM layers are being used in the network. The number of layers equals to the size of hidden_layers_sizes array. Also from the code above, the number of neurons in each layers are 1000.

To speed up the network to make comparison, I decreased the epoch even smaller. 

```python
281  def test_DBN(finetune_lr=0.1, pretraining_epochs=2,
282              pretrain_lr=0.01, k=1, training_epochs=20,
283              dataset='mnist.pkl.gz', batch_size=10)
```


