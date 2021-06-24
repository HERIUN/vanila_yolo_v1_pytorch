import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
import numpy as np

from darknet import DarkNet

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
        self.features = features # yolov1 uses pretrained darkent layers..
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

    def load_weights(self, weightfile):

        fp = open(weightfile, 'rb')
        weight = np.fromfile(fp, dtype=np.float32)
        print(f"{weightfile} has {weight.size} weight & bias")
        fp.close()

        ptr = 0
        sum_b = 0
        sum_w = 0 
        for layer in self.features:
            if isinstance(layer, nn.modules.conv.Conv2d):
                num_w = layer.weight.numel()
                num_b = layer.bias.numel()
                sum_b += num_b
                print(f"loading conv weight :{layer}, # of weights : {num_w}, # of bias : {num_b}")
                layer.bias.data.copy_(torch.from_numpy(weight[ptr: ptr+num_b]).view_as(layer.bias.data))
                ptr += num_b
                layer.weight.data.copy_(torch.from_numpy(weight[ptr: ptr+num_w]).view_as(layer.weight.data))
                ptr += num_w
                print(f"{weight.size - ptr} weight & bias remain")
        
        for layer in self.fc:
            if isinstance(layer, nn.modules.Linear):
                num_w = layer.weight.numel()
                num_b = layer.bias.numel()
                print(f"loading fc weight :{layer}, # of weights : {num_w}, # of bias : {num_b}")
                layer.bias.data.copy_(torch.from_numpy(weight[ptr: ptr+num_b]).view_as(layer.bias.data))
                ptr += num_b
                layer.weight.data.copy_(torch.from_numpy(weight[ptr: ptr+num_w]).view_as(layer.weight.data))
                ptr += num_w
                print(f"{weight.size - ptr} weight & bias remain")

        return True

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def test():
    one_mini_batch = torch.rand(1, 3, 448, 448)

    cfg_path = "/home/lee/workspace/vanila_yolo_v1_pytorch/weight_cfg/extraction.conv.cfg"

    weight_path = "/home/lee/workspace/vanila_yolo_v1_pytorch/weight_cfg/extraction.conv.weights"

    darknet = DarkNet(cfg_path)
    darknet.load_weights(weight_path)
    yolo = YOLOv1(darknet.features)

    print(yolo)

    print(1)
    output = yolo(one_mini_batch)
    print(output.size()) # must be torch.Size([20, 7, 7, 30])

if __name__ == '__main__':
    test()