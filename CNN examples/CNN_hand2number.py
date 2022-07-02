import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from cnn_utils import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.random.seed(1)

torch.set_default_dtype(torch.float64)

class CNN(nn.Module):
    def __init__(self, Inp_ch, Out_ch):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(Inp_ch, 8, (4,4), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((8,8), stride=(8,8), padding=(0,0)),
            
            nn.Conv2d(8, 16, (2,2), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4,4), stride=(4,4), padding=(0,0)),
            
            nn.Flatten(),
            nn.Linear(16, Out_ch)
            )
    def forward(self, x):
        #CrossEntropyLoss automatically applies the softmax function on the output of the network
        return self.layers(x)


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
            loss = CEF_loss(Y_out,minibatch_Y) + Lambda* sum([torch.sum(p**2) for p in model.parameters()])
            loss.backward()
            optimizer.step()
            
            epoch_cost += loss.item() / num_minibatches
        if print_cost == True and epoch % 20 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 10 == 0:
            costs.append(epoch_cost)
    
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
   
    return model


# Loading the data (signs)
X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

# Example of a picture
index = 3
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

(m, h, w, c) = X_train.shape
X_train= X_train.reshape((m, c, h, w))
(m, h, w, c) = X_test.shape
X_test = X_test.reshape((m, c, h, w))
"""Training the model"""
trained_model = model(torch.tensor(X_train), torch.LongTensor(Y_train.flatten()), 
                      torch.tensor(X_test), torch.LongTensor(Y_test.flatten()), 
                      classes,  learning_rate = 0.001,  num_epochs = 1500, minibatch_size = 16)

"""Prediction"""
trained_model.eval()
Y_out = torch.argmax(trained_model(torch.tensor(X_test)), dim=1)
accuracy = (1-torch.sum(Y_out!=torch.LongTensor(Y_test.flatten()))/Y_out.shape[0])*100
print('accuracy is: %f'%accuracy)