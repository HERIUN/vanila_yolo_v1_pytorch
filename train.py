
import torch
from torch.utils.data import DataLoader
import numpy as np
import math

from voc_dataset import VOCDataset
from model import YOLOv1
from loss import Loss

import os
import os.path as pth

classes = ['person', # Person
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', # Animal
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', # Vehicle
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor' # Indoor
           ]

# annot_dir = '/home/lee/workspace/yolo_v1_pytorch/VOCdevkit/VOC2012/Annotations'

image_dir = '/home/lee/workspace/yolo_v1_pytorch/VOC2012/JPEGImages'
label_txt = '/home/lee/workspace/yolo_v1_pytorch/voc2012.txt'
# convert_annot_dir = '/home/lee/workspace/yolo_v1_pytorch/voc_to_yolo '
# class_names_file_dir = '/home/lee/workspace/yolo_v1_pytorch/voc.names'

# voc_xml_to_txt_command = f"python3 voc_to_yolo.py --datasets VOC --img_path {image_dir} --label {annot_dir} --convert_output_path {convert_annot_dir} --img_type "".jpg"" --manifest_path ./ --cls_list_file {class_names_file_dir}"
# os.system(voc_xml_to_txt_command)


# Training hyper parameters.
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 16 ## 원래는 64인데, 1070 8GB에서는 감당이 안되서, 16으로 줄이겠습니당.


# Learning rate scheduling.
def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0: # for the first epochs. slowly raise 10^-3 to 10^-2
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1: # 10^-2 for 75 epochs
        lr = base_lr
    elif epoch == 75: # 10^-3 for 30 epochs 
        lr = 0.001
    elif epoch == 105: # 10^-4 for 30 epochs
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


yolo = YOLOv1()
yolo.cuda()

criterion = Loss()
optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay) # SGD or Adam ?

train_dataset = VOCDataset(image_dir, label_txt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

for epoch in range(1):
    
    yolo.train()

    for i, (imgs, targets) in enumerate(train_loader):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, num_epochs))

        update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
        lr = get_lr(optimizer)

        imgs = imgs.type(torch.FloatTensor)
        imgs, targets = imgs.cuda(), targets.cuda() # torch.cuda.DoubleTensor.. to torch.cuda.FloatTensor

        #Forward
        preds = yolo(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

