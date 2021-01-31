"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

import torchvision.models as models

class SegmentationNN(pl.LightningModule):
    
    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        alexnet_encoder = models.alexnet(pretrained=True)
        
        self.encoder = alexnet_encoder.features
        
        self.encoder_par = {}
        for i, param in enumerate(self.encoder.parameters()):
            param.requires_grad = False
            self.encoder_par[i] = param
            
            
        print("1st Conv", self.encoder_par[0].shape)
        print("2nd Conv", self.encoder_par[2].shape)
        print("3rd Conv", self.encoder_par[4].shape)
        print("4th Conv", self.encoder_par[6].shape)
        print("5th Conv", self.encoder_par[8].shape)

        #print("encoder: ", encoder)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), #12*12
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 312, kernel_size=3, stride=1, padding=1), #12*12
            nn.BatchNorm2d(312),
            nn.ReLU(),
            nn.ConvTranspose2d(312, 192, kernel_size=3, stride=1, padding=1), #12*12
            nn.BatchNorm2d(192),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 24 * 24
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=5), # 40 * 40
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 80*80
            nn.ConvTranspose2d(64, num_classes, kernel_size=7, stride=3, padding=2), #240 * 240
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
        #print("decoder: ", decoder)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        # Get skip connection features
        self.featuremap1 = self.encoder_par[0](x)
        print("feature1:", self.featuremap1)

        x = self.encoder(x)
        x = self.decoder(x)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
