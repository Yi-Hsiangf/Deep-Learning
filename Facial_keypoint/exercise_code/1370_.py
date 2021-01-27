"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representinsssg each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=0), #94*94
            nn.Conv2d(16, 16, 3, stride=1, padding=0), #92*92
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #46*46 
            nn.Dropout2d(p=0.1)
            
            nn.Conv2d(16, 32, 3, stride=1, padding=0), #44 * 44 
            nn.Conv2d(32, 32, 3, stride=1, padding=0), #42 * 42 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 21*21
            nn.Dropout2d(p=0.1)
            
            nn.Conv2d(32, 64, 5, stride=1, padding=0), #17*17
            nn.Conv2d(64, 64, 5, stride=1, padding=0), #13*13
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 6*6
            nn.Dropout2d(p=0.1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(0.4),
            nn.Dropout(dropout)
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(0.4),
            nn.Dropout(dropout)
            nn.Linear(500, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
            nn.Linear(128, 30),
            nn.BatchNorm1d(30),
            nn.Tanh()
        )
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.cnn(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
