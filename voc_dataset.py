import torch
from torch.nn.functional import cosine_embedding_loss
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import random
import numpy as np
import os
import cv2

class VOCDataset(Dataset):

    def __init__(self, image_dir, label_txt, image_size=448, grid_size=7, num_bboxes=2, num_classes=20):
        self.image_size = image_size

        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes
        self.to_tensor = transforms.ToTensor()

        self.img_paths, self.boxes, self.labels = [], [], []

        with open(label_txt) as f:
            lines = f.readlines()
        
        for line in lines: # 각 라인당 fname, [x1, y1, x2, y2, class]_obj1, [...]_obj2...
            splitted = line.strip().split()

            fname = splitted[0]
            path = os.path.join(image_dir,fname)
            self.img_paths.append(path)

            num_boxes = (len(splitted)-1) //5
            box, label = [], []
            for i in range(num_boxes):
                x1 = float(splitted[5*i + 1])
                y1 = float(splitted[5*i + 2])
                x2 = float(splitted[5*i + 3])
                y2 = float(splitted[5*i + 4])
                c  =   int(splitted[5*i + 5])
                box.append([x1,y1,x2,y2])
                label.append(c)
            self.boxes.append(torch.Tensor(box)) # [tensor([x1,y1,x2,y2], [x1,y1,x2,y2], ...), tensor(...)]
            self.labels.append(torch.LongTensor(label))

        self.num_samples = len(self.img_paths)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path) # (h, w, 3). 3 = (B,G,R)
        boxes = self.boxes[idx].clone() # [n, 4] ex. tensor([[x1, y1, x2, y2]_obj1])
        labels = self.labels[idx].clone() # [n,]

        img, boxes = self.random_scale(img,boxes)
        img, boxes, labels = self.random_translation(img, boxes, labels)
        img = self.random_exposure(img)
        img = self.random_saturation(img)

        # For debug.
        debug_dir = 'tmp/voc_tta'
        os.makedirs(debug_dir, exist_ok=True)
        img_show = img.copy()
        box_show = boxes.numpy().reshape(-1)
        n = len(box_show) // 4
        for b in range(n):
            pt1 = (int(box_show[4*b + 0]), int(box_show[4*b + 1]))
            pt2 = (int(box_show[4*b + 2]), int(box_show[4*b + 3]))
            cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0,255,0), thickness=1)
        cv2.imwrite(os.path.join(debug_dir, 'test_{}.jpg'.format(idx)), img_show)

        h, w, _ = img.shape
        boxes /= torch.Tensor([[w,h,w,h]])#.expand_as(boxes) # normalize(x1,y1,x2,y2) w.r.t. image width,height
        target = self.encode(boxes, labels) # [S, S, 5 x B + C]

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0 # normalize from 0 to 1
        img = self.to_tensor(img)

        return img, target

    def encode(self, boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: tensor([[x1, y1, x2, y2]_obj1, ...]) normalized from 0.0 to 1.0 w.r.t image width/height. ex. tensor([[0.5, 0.3, 0.6, 0.8], ...])
            labels: tensor([c_obj1, c_obj2, ...])   ex. tensor([1, 15, 19, ...])
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C # 30

        target = torch.zeros(S, S, N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]
        
        for bi in range(boxes.size(0)):
            xy, wh, label = boxes_xy[bi], boxes_wh[bi], int(labels[bi])
            ij = (xy / cell_size).ceil() - 1.0 # 해당 박스의 center가 어떤 grid에 속하는지 알아내기 위함. 1을 빼주는 이유는 index가 0부터 시작이므로
            i, j = int(ij[0]), int(ij[1]) # grid index
            x0y0 = ij * cell_size # 해당 그리드의 왼쪽 상단의 좌표
            xy_normalized_grid = (xy - x0y0) / cell_size # 해당 그리드의 넓이,높이를 1로 쳤을때, center x,y의 좌표(0~1)
            for k in range(B):
                target[i, j, 5*k   : 5*k+2] = xy_normalized_grid
                target[i, j, 5*k+2 : 5*k+4] = wh # check paper section 2.Unified Detection, 4th paragraph
                target[i, j, 5*k+4        ] = 1.0 # confidence score. 물체가 있을 확률(Pr(Obj)) * IOU_truth/pred
            target[i, j, 5*B + label      ] = 1.0 # class probability. Pr(Class_i|Object)

        return target

    def random_scale(self, img, boxes):
        if random.random() < 0.5:
            return img, boxes
        
        scale = random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        
        img = cv2.resize(img, dsize=(int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR) # why h is not scaling
        scale_tensor = torch.FloatTensor([[scale, scale, scale, scale]]).expand_as(boxes) # duplicate N object(bbox)
        boxes = boxes * scale_tensor

        return img, boxes

    def random_translation(self, img, boxes, labels):
        """
            이미지를 가로,세로 최대 20%까지 평행이동. 만약 박스가 이미지 밖으로 나갈 경우, 이미지의 끝에 맞춰짐.
        """
        if random.random() < 0.5:
            return img, boxes, labels
        
        h, w, c = img.shape
        img_out = np.zeros((h, w, c), dtype=img.dtype)
        center = (boxes[:, 2:] + boxes[:, :2]) / 2.0

        dx = random.uniform(-w*0.2, w*0.2)
        dy = random.uniform(-h*0.2, h*0.2)
        dx, dy = int(dx), int(dy)

        if dx >= 0 and dy >= 0:
            img_out[dy:, dx:] = img[:h-dy, :w-dx]
        elif dx >= 0 and dy < 0:
            img_out[:h+dy, dx:] = img[-dy:, :w-dx]
        elif dx < 0 and dy >= 0:
            img_out[dy:, :w+dx] = img[:h-dy, -dx:]
        elif dx < 0 and dy < 0:
            img_out[:h+dy, :w+dx] = img[-dy:, -dx:]

        center = center + torch.FloatTensor([[dx, dy]])#.expand_as(center) # [n,2]
        # if center of boxes is within images, then mask_x,y is true
        mask_x = (center[:, 0] >= 0) & (center[:, 0] < w) # [n,] ex. tensor([True, True]) : two boxes
        mask_y = (center[:, 1] >= 0) & (center[:, 1] < h) # [n,]
        mask = (mask_x & mask_y).view(-1,1) #[n, 1] ex. tensor([[True],[True]])
        boxes_out = boxes[mask.expand_as(boxes)].view(-1, 4) # [m, 4]

        if len(boxes_out) == 0:
            return img, boxes, labels
        shift = torch.FloatTensor([[dx, dy, dx, dy]]).expand_as(boxes_out) # [m, 4]

        boxes_out = boxes_out + shift
        boxes_out[:, 0] = boxes_out[:, 0].clamp_(min=0, max=w)
        boxes_out[:, 2] = boxes_out[:, 2].clamp_(min=0, max=w)
        boxes_out[:, 1] = boxes_out[:, 1].clamp_(min=0, max=h)
        boxes_out[:, 3] = boxes_out[:, 3].clamp_(min=0, max=h)

        labels_out = labels[mask.view(-1)]

        return img_out, boxes_out, labels_out

    def random_exposure(self, bgr):
        """
            same as random_brightness
        """
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr
        
    def random_saturation(self, bgr):
        """
            채도 조절
        """
        if random.random() < 0.5:
            return bgr

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(0.5, 1.5)
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

def test():
    from torch.utils.data import DataLoader

    image_dir = '/home/lee/workspace/vanila_yolo_v1_pytorch/VOC2012/JPEGImages'
    label_txt = '/home/lee/workspace/vanila_yolo_v1_pytorch/voc2012.txt'

    dataset = VOCDataset(image_dir, label_txt)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iter = iter(data_loader)
    for i in range(10):
        img, target = next(data_iter)
        print(img.size(), target.size())
        # img's shape must be N,C,H,W = (1, 3, 448, 448)
        # target's size must be N,S,S,5*B+C = (1, 7, 7, 30)

if __name__ == '__main__':
    test()