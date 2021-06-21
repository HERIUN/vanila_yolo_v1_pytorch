import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax

class YOLOv1(nn.Module):
    """
    input image size : 448*448
    output size : 7*7 * 30
        7 means grid
        30 means 2 Bbox information(x,y,w,h,confidence) + class probability (20)
    """
    def __init__(self, features):
        super(YOLOv1, self).__init__()

        # Make layers
        self.features = features
        self.features2 = self._make_conv_layers()
        self.fc = self._make_fc_layers()

    def forward(self, x): 
        x = self.features(x)
        x = self.features2(x)
        x = self.fc(x)
        x = x.view(-1, 7, 7, 5*2 + 20)
        return x

    def _make_conv_layers(self): # 4 Convs. 
        conv = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        return conv

    def _make_fc_layers(self): ## 2 Fully connected
        fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=7*7*1024, out_features=4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5), # defalut inplace = False
            nn.Linear(in_features=4096, out_features=(7*7*(5*2+20))), # 7 : grid cell, 5 : bbox's value(x,y,w,h,conf), 2 : number of bboxs, 20 : number of classes
            # nn.Softmax() or nn.Sigmoid()
        )
        return fc

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Squeeze(nn.Module): ## 나중에 안쓰면 빼자.

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


def test():
    from torch.autograd import Variable
    image = torch.rand(20, 3, 448, 448)

    yolo = YOLOv1()

    output = yolo(image)
    print(output.size()) # must be torch.Size([20, 7, 7, 30])

if __name__ == '__main__':
    test()