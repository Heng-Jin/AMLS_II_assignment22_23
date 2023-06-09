import math
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, d=56, s=12, m=4):
        '''
        SRCNN network structure and weight initialization
        Args:
            scale_factor: enlargement scale of images, HR size/ LR size
            num_channels: input images of channels, default channel num is 1, only 'Y' of YCbCr will be loaded
            d: number of channels in expanding
            s: number of channels in mapping
            m: number of mapping blocks
        '''
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        '''

        Weights initialization

        '''
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        '''
        FSRCNN forward
        Args:
            x: torch.tensor

        Returns: torch.tensor

        '''
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class FSRCNN_Residual(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, d=56, s=12, m=4):
        '''
        SRCNN network structure and weight initialization
        Args:
            scale_factor: enlargement scale of images, HR size/ LR size
            num_channels: input images of channels, default channel num is 1, only 'Y' of YCbCr will be loaded
            d: number of channels in expanding
            s: number of channels in mapping
            m: number of mapping blocks
        '''
        super(FSRCNN_Residual, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        '''

        Weights initialization

        '''
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        '''
        Residual FSRCNN forward, skip connection is implemented by x+=residual
        Args:
            x: torch.tensor

        Returns: torch.tensor

        '''
        x = self.first_part(x)
        residual = x
        x = self.mid_part(x)
        x += residual
        x = self.last_part(x)
        return x