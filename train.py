import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

from torchsummary import summary

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

# create a class for the model
class mnistNet(nn.Module):
    def __init__(self,printtoggle=False):
      super().__init__()

      ### convolution layers
      self.conv1 = nn.Conv2d( 1,10,kernel_size=5,stride=1,padding=1)
      # size: np.floor( (28+2*1-5)/1 )+1 = 26/2 = 13

      self.conv2 = nn.Conv2d(10,20,kernel_size=5,stride=1,padding=1)
      # size: np.floor( (13+2*1-5)/1 )+1 = 11/2 = 5 (/2 b/c maxpool)

      # compute the number of units in FClayer (number of outputs of conv2)
      expectSize = np.floor( (5+2*0-1)/1 ) + 1 # fc1 layer has no padding or kernel, so set to 0/1
      expectSize = 20*int(expectSize**2)

      ### fully-connected layer
      self.fc1 = nn.Linear(expectSize,50)

      ### output layer
      self.out = nn.Linear(50,10)

      self.print = printtoggle

    # forward pass
    def forward(self,x):

      print(f'Input: {x.shape}') if self.print else None

      # convolution -> maxpool -> relu
      x = F.relu(F.max_pool2d(self.conv1(x),2))
      print(f'Layer conv1/pool1: {x.shape}') if self.print else None

      # and again: convolution -> maxpool -> relu
      x = F.relu(F.max_pool2d(self.conv2(x),2))
      print(f'Layer conv2/pool2: {x.shape}') if self.print else None

      nUnits = x.shape.numel()/x.shape[0]
      x = x.view(-1,int(nUnits))
      if self.print: print(f'Vectorize: {x.shape}')

      x = F.relu(self.fc1(x))
      if self.print: print(f'Layer fc1: {x.shape}')
      x = self.out(x)
      if self.print: print(f'Layer out: {x.shape}')

      return x

def main():

  root = "./data"
  transform = transforms.ToTensor()
  train_data = datasets.MNIST(root=root, train=True, download=True, transform=transform)
  test_data = datasets.MNIST(root=root, train=False, download=True, transform=transform)

  # translating into Dataloader objects
  batch_size = 64
  train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

  # # shape
  train_loader.dataset.data.shape
    
  def createTheMNISTNet(printtoggle=False):

    net = mnistNet(printtoggle)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=.001)

    return net,lossfun,optimizer

  # # sanity check
  net,lossfun,optimizer = createTheMNISTNet(True)
  X,y = next(iter(train_loader))
  yHat = net(X)

  # # size of model
  print(' ')
  print(yHat.shape)
  print(y.shape)

  # # computing loss
  loss = lossfun(yHat,y)
  print(' ')
  print('Loss:')
  print(loss)

  # # count the total number of parameters in the model
  summary(net,(1,28,28))


  # a function that trains the model
  def function2trainTheModel():

    numepochs = 1

    net,lossfun,optimizer = createTheMNISTNet()

    # initialize losses
    losses    = torch.zeros(numepochs)
    trainAcc  = []
    testAcc   = []


    # loop over epochs
    for epochi in range(numepochs):

      net.train()
      batchAcc  = []
      batchLoss = []
      for X,y in train_loader:

        # forward pass and loss
        yHat = net(X)
        loss = lossfun(yHat,y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchLoss.append(loss.item())

        # compute accuracy
        matches = torch.argmax(yHat,axis=1) == y
        matchesNumeric = matches.float()
        accuracyPct = 100*torch.mean(matchesNumeric)
        batchAcc.append( accuracyPct )

      trainAcc.append( np.mean(batchAcc) )

      losses[epochi] = np.mean(batchLoss)

      # test accuracy
      net.eval()
      X,y = next(iter(test_loader))
      with torch.no_grad():
        yHat = net(X)

      testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()))

    # end epochs

    return trainAcc,testAcc,losses,net

  # # estimate time to train for 100 epochs on google colab GPU 55 mins (not needed to train for 100 epochs though)
  trainAcc,testAcc,losses,net = function2trainTheModel()
  torch.save(net.state_dict(),'mnist_model.pth')

  # # plotting accuracy and loss
  fig,ax = plt.subplots(1,2,figsize=(16,5))

  ax[0].plot(losses,'s-')
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Loss')
  ax[0].set_title('Model loss')

  ax[1].plot(trainAcc,'s-',label='Train')
  ax[1].plot(testAcc,'o-',label='Test')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Accuracy (%)')
  ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
  ax[1].legend()

  plt.show()


if __name__ == "__main__":
    main()