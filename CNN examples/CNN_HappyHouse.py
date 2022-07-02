import numpy as np
import h5py
import torch
from torch import nn, optim
from cnn_utils import *
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.random.seed(1)

torch.set_default_dtype(torch.float64)

"""
If your happyModel() function worked, you should have observed much better than 
random-guessing (50%) accuracy on the train and test sets. To pass this assignment, 
you have to get at least 75% accuracy.

To give you a point of comparison, our model gets around 95% test accuracy in 40 
epochs (and 99% train accuracy) with a mini batch size of 16 and "adam" optimizer. 
But our model gets decent accuracy after just 2-5 epochs, so if you're comparing 
different models you can also train a variety of models on just a few epochs and 
see how they compare.

If you have not yet achieved 75% accuracy, here're some things you can play around 
with to try to achieve it:

Try using blocks of CONV->BATCHNORM->RELU such as:
X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)
until your height and width dimensions are quite low and your number of channels 
quite large (≈32 for example). You are encoding useful information in a volume with 
a lot of channels. You can then flatten the volume and use a fully-connected layer.
You can use MAXPOOL after such blocks. It will help you lower the dimension in 
height and width.

Change your optimizer. We find Adam works well.
If the model is struggling to run and you get memory issues, lower your 
batch_size (12 is usually a good compromise)

Run on more epochs, until you see the train accuracy plateauing.
Even if you have achieved 75% accuracy, please feel free to keep playing with your
model to try to get even better results.

Note: If you perform hyperparameter tuning on your model, the test set actually becomes a dev set, and your model might end up overfitting to the test (dev) set. But just for the purpose of this assignment, we won't worry about that here.
"""


"""
Exercise: Implement a HappyModel(). This assignment is more open-ended than most. 
We suggest that you start by implementing a model using the architecture we suggest, 
and run through the rest of this assignment using that as your initial model. But after 
that, come back and take initiative to try out other model architectures. For example, 
you might take inspiration from the model above, but then vary the network architecture 
and hyperparameters however you wish. You can also use other functions such as 
AveragePooling2D(), GlobalMaxPooling2D(), Dropout().
"""


def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


class CNN(nn.Module):
    def __init__(self, Inp_ch, Out_ch):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(Inp_ch, 32, (7,7), stride=(3,3), padding=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)),
            
            nn.Flatten(),
            nn.Linear(3200, Out_ch)
            )
    def forward(self, x):
        return self.layers(x)
# m = nn.Sequential(
#     nn.Conv2d(3, 32, (7,7), stride=(3,3), padding=(3,3)),
#     nn.BatchNorm2d(32),
#     nn.ReLU(),
#     nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)),
#     nn.Flatten()
#       )
# print(m(torch.tensor(X_train[0:1])).shape)

def model(X_train, Y_train, X_test, Y_test, classes, learning_rate = 0.001,
          num_epochs = 1500, minibatch_size = 32, Lambda = 0.000001, print_cost = True):
    
    (m, c, h, w) = X_train.shape
    n_y = len(classes)  
    
    costs = []
    
    model = CNN(c, n_y)
    CEF_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    seed = 0
    for epoch in range(num_epochs):
        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
        
        for minibatch_X, minibatch_Y in minibatches:
            optimizer.zero_grad()
            Y_out = model.forward(minibatch_X)
            loss = CEF_loss(Y_out,minibatch_Y) #+ Lambda* sum([torch.sum(p**2) for p in model.parameters()])
            loss.backward()
            optimizer.step()
            
            epoch_cost += loss.item() / num_minibatches
        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)
    
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()    
    return model


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize image vectors
(n, h, w, c) = X_train_orig.shape
X_train = X_train_orig.reshape(n, c, h, w)/255.
(n, h, w, c) = X_test_orig.shape
X_test = X_test_orig.reshape(n, c, h, w)/255.
# Reshape
Y_train = Y_train_orig.flatten()
Y_test = Y_test_orig.flatten()
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

trained_model = model(torch.tensor(X_train), torch.LongTensor(Y_train.flatten()), 
                      torch.tensor(X_test), torch.LongTensor(Y_test.flatten()), 
                      classes,  learning_rate = 0.0001,  num_epochs = 1500, minibatch_size = 16)


"""Prediction"""
trained_model.eval()
Y_out = torch.argmax(trained_model(torch.tensor(X_test)), dim=1)
accuracy = (1-torch.sum(Y_out!=torch.LongTensor(Y_test.flatten()))/Y_out.shape[0])*100
print('accuracy is: %f'%accuracy)









