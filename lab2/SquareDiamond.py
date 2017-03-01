############################################CREATE TRANING###############################################################import os
import os
import caffe
import numpy as np
import matplotlib.pyplot as plt
############################################Pre Processing Data##############################################################
my_root = '/home/martin/Desktop/Caffe_Files/caffe_example/SquareDiamond'
os.chdir(my_root)


#######################################Train the Network with the Solver######################################################

caffe.set_device(0)
caffe.set_mode_gpu()

solver =  caffe.SGDSolver('Square_diamond_solver.prototxt')

niter =2000
test_interval = 100
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')

    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc



#----------------------------------------------------------------------------------------------
###########################Plotting Intermediate Layers, Weight################################
#---------------------------------------Define Functions---------------------------------------

def vis_square_f(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data,cmap='Greys',interpolation='nearest');
    plt.axis('off')
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Feature maps Functions---------------------------------
plt.figure(1)
plt.semilogy(np.arange(niter), train_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')

plt.figure(2)
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')

#----------------------------------------------------------------------------------------------
#------------------------------Plot All Feature maps Functions---------------------------------
net = solver.net
f1_0 = net.blobs['conv1'].data[0, :2]
plt.figure(3)
vis_square_f(f1_0)
plt.title('Feature Maps for Conv1')
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Kernels for Conv1---------------------------------------
nrows = 2                                   # Number of Rows
ncols = 2                                   # Number of Columbs
ker_size = 3                                # Kernel Size
Zero_c= np.zeros((ker_size,1))              # Create np.array of zeros
Zero_r = np.zeros((1,ker_size+1))
M= np.array([]).reshape(0,ncols*(ker_size+1))

for i in range(nrows):
    N = np.array([]).reshape((ker_size+1),0)

    for j in range(ncols):
        All_kernel = net.params['conv1'][0].data[j + i * ncols][0]

        All_kernel = np.matrix(All_kernel)
        All_kernel = np.concatenate((All_kernel,Zero_c),axis=1)
        All_kernel = np.concatenate((All_kernel, Zero_r), axis=0)
        N = np.concatenate((N,All_kernel),axis=1)
    M = np.concatenate((M,N),axis=0)

plt.figure(4)
plt.imshow(M, cmap='Greys',  interpolation='nearest')
plt.title('All Kernels for Conv1')
#----------------------------------------------------------------------------------------------
#------------------------------Plot one Kernels for Conv1--------------------------------------
ker1_0 = net.params['conv1'][0].data[0]      #net.params['conv1'][0] is reffering to Weights
ker1_0 = np.matrix(ker1_0)
plt.figure(5)
plt.imshow(ker1_0, cmap='Greys',  interpolation='nearest')
plt.title('One Kernels for Conv1')
plt.show()
#----------------------------------------------------------------------------------------------
#---------------------------Print Shape ans Sizes for all Layers--------------------------------

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
