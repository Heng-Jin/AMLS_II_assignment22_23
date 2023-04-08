import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, padding=False, num_channels=1):
        '''
        SRCNN network structure and weight initialization
        Args:
            padding:
            num_channels: input images of channels, default channel num is 1, only 'Y' of YCbCr will be loaded
        '''
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=9, padding=4*int(padding), padding_mode='replicate'),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0),  # n1 * 1 * 1 * n2
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2*int(padding), padding_mode='replicate')

    def forward(self, x):
        '''
        FSRCNN forward
        Args:
            x: torch.tensor

        Returns: torch.tensor

        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def init_weights(self):
        '''

        Weights initialization

        '''
        for L in self.conv1:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=0.001)
                L.bias.data.zero_()
        for L in self.conv2:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=0.001)
                L.bias.data.zero_()
        self.conv3.weight.data.normal_(mean=0.0, std=0.001)
        self.conv3.bias.data.zero_()