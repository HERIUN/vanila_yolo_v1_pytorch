import torch
import torch.nn as nn
import torch.nn.functional as F

class DarkNet(nn.Module):
    def __init__(self, init_weight=True):
        super(DarkNet, self).__init__()

        # Make layers
        self.features = self._make_conv_layers()
        self.fc = self._make_fc_layers()

        # Initialize weights
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

    def _make_conv_layers(self): ## 20 Convs, used for pretrained by IMAGE Net 1000 class
        conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), # padding=3 so, output is 224.
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(192, 128, 1), ## kernel size = 1 이므로 padding = 0(defalut)
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 256, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return conv

    def _make_fc_layers(self):
        fc = nn.Sequential(
            nn.AvgPool2d(7),
            Squeeze(),
            nn.Linear(1024, 1000)
        )
        return fc

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()