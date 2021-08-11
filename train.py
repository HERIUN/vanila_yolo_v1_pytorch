
import torch
from torch.utils.data import DataLoader
import numpy as np
import math

from voc_dataset import VOCDataset
from darknet import DarkNet
from yolo_v1 import YOLOv1
from loss import Loss

import os
import os.path as pth
# import wandb
from datetime import datetime
from tensorboardX import SummaryWriter


def main():
    # wandb.login()

    use_gpu = torch.cuda.is_available()
    assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA.'
    print('CUDA current_device: {}'.format(torch.cuda.current_device()))
    print('CUDA device_count: {}'.format(torch.cuda.device_count()))

    classes = ['person', # Person
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', # Animal
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', # Vehicle
            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor' # Indoor
            ]


    image_dir = '/home/kang/workspace/vanila_yolo_v1_pytorch/VOC2012/JPEGImages'
    label_txt = '/home/kang/workspace/vanila_yolo_v1_pytorch/voc2012.txt'

    # run = wandb.init(project="vanila_yolov1", 
    #                 config={
    #                     "optimizer" : "SGD with momentum",
    #                     "batch_size :" : "32",
    #                     "train_data :" : "VOC2007"})

    # Training hyper parameters.
    init_lr = 0.001
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 5.0e-4
    num_epochs = 135
    batch_size = 64  ## 원래는 64인데, gtx1070 8GB에서는 감당이 안되서, 16으로 줄이겠습니당.

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

    # Load Pretrained Network
    cfg_path = "/home/kang/workspace/vanila_yolo_v1_pytorch/weight_cfg/extraction.conv.cfg"
    weight_path = "/home/kang/workspace/vanila_yolo_v1_pytorch/weight_cfg/extraction.conv.weights"
    darknet = DarkNet(cfg_path)
    darknet.load_weights(weight_path)

    yolo = YOLOv1(darknet.features)
    yolo.cuda()

    criterion = Loss()
    optimizer = torch.optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay) # SGD or Adam ?

    train_dataset = VOCDataset(image_dir, label_txt)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print('Number of training images :', len(train_dataset))

    # Open TensorBoardX summary writer
    log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('results/yolo', log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop.
    logfile = open(os.path.join(log_dir, 'log.txt'), 'w')
    best_val_loss = np.inf

    for epoch in range(5):
        print('\n')
        print('Starting epoch {} / {}'.format(epoch, num_epochs))

        yolo.train()
        total_loss = 0.0
        total_batch = 0

        for i, (imgs, targets) in enumerate(train_loader):

            update_lr(optimizer, epoch, float(i) / float(len(train_loader) - 1))
            lr = get_lr(optimizer)

            batch_size_this_iter = imgs.size(0)
            imgs = imgs.type(torch.FloatTensor)
            imgs, targets = imgs.cuda(), targets.cuda() # torch.cuda.DoubleTensor.. to torch.cuda.FloatTensor

            # Forward
            preds = yolo(imgs)
            loss = criterion(preds, targets)
            loss_this_iter = loss.item()
            total_loss += loss_this_iter * batch_size_this_iter
            total_batch += batch_size_this_iter

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print current loss.
            if i % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Loss: %.4f, Average Loss: %.4f'
                % (epoch, num_epochs, i, len(train_loader), lr, loss_this_iter, total_loss / float(total_batch)))
            
            # wandb.log({
            #     "Epoch": epoch,
            #     "Iter": i/len(train_loader),
            #     "LR": lr,
            #     "Loss": loss_this_iter,
            #     "Average Loss": total_loss / float(total_batch)
            # })


    # log_dir = datetime.now().strftime('%b%d_%H-%M-%S')
    # log_dir = os.path.join('results/yolo', log_dir)
    # torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_latest.pth'))

if __name__ == '__main__':
    main()

