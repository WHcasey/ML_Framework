# this is a simple Convolutional Neural Network built using PyTorch
from torch import nn


class NeuralNetwork(nn.Module):                                           # create convolutional neural network class
    def __init__(self):                                                   # initiate the class
        super(NeuralNetwork, self).__init__()                             # import exising items from torch class
        self.network = nn.Sequential(                                     # create a Sequential neural network
            nn.Conv2d(1, 32, kernel_size=10, padding=1),                  # add first convolution layer
            nn.ReLU(),                                                    # activate first layer
            nn.Conv2d(32, 64, kernel_size=10, stride=2, padding=1),       # second convolution layer
            nn.ReLU(),                                                    # activate second layer
            nn.MaxPool2d(2,2),                                            # peform pooling on convolutions
        
            nn.Conv2d(64, 128, kernel_size=10, stride=2, padding=1),      # third convolution layer
            nn.ReLU(),                                                    # activate third layer
            nn.Conv2d(128, 128, kernel_size=10, stride=2, padding=1),     # fourth convolution layer
            nn.ReLU(),                                                    # activate fourth layer
            nn.MaxPool2d(2,2),                                            # perform pooling on convolutions
        
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),      # fifth convolution layer
            nn.ReLU(),                                                    # activate fifth layer
            nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=1),      # sixth convolution layer
            nn.ReLU(),                                                    # activate sixth layer
            nn.MaxPool2d(2,2),                                            # perform pooling on convolutions
        
            nn.Flatten(),                                                 # flatten convolutions into a 1xN array
            nn.Linear(512, 1024),                                         # feed into dense network
            nn.ReLU(),                                                    # activate neurons
            nn.Linear(1024, 512),                                         # feed into dense network
            nn.ReLU(),                                                    # activate neurons
            nn.Linear(512,27),                                            # feed into dense network
            nn.Sigmoid())                                                 # perform final activation (all between 0 and 1)
            
    def forward(self, x):            # create forward function
        return self.network(x)       # return network
        
