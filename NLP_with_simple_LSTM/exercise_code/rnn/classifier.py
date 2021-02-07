import pickle
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .rnn_nn import *
from .base_classifier import *


class RNN_Classifier(Base_Classifier):
    
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################
        self.model_linear = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/2), int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), classes),
        )

        self.rnn = nn.RNN(input_size, hidden_size)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


    def forward(self, x):
    ############################################################################
    #  TODO: Perform the forward pass                                          #
    ############################################################################   

        output, _ = self.rnn(x) # 28 (seq_len) x (batch_size) x (hidden_size): 28 x batch_size x 128
        output = output[-1]  # use last output from RNN (has seen all sequences)
        x = self.model_linear(output)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return x


class LSTM_Classifier(Base_Classifier):

    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.model_linear = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.BatchNorm1d(int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), int(hidden_size / 2)),
            nn.BatchNorm1d(int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), classes),
        )

        self.rnn = nn.LSTM(input_size, hidden_size)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


    def forward(self, x):

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################    

        output, _ = self.rnn(x)  # 28 (seq_len) x (batch_size) x (hidden_size): 28 x batch_size x 128
        output = output[-1]  # use last output from RNN (has seen all sequences)
        x = self.model_linear(output)

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return x
