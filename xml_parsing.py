import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


txt_file = open('voc2007.txt', 'w')
Annotations_dir = '/home/kang/workspace/vanila_yolo_v1_pytorch/VOC2007/Annotations/'
xml_files = os.listdir(Annotations_dir)

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}

        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

count = 0
for xml_file in xml_files:
    count += 1

    image_path = xml_file.split('.')[0] + '.jpg'
    results = parse_rec(Annotations_dir + xml_file)

    txt_file.write(image_path)
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_idx = VOC_CLASSES.index(class_name)
        txt_file.write(' ' +
                       str(bbox[0]) +
                       ' ' +
                       str(bbox[1]) +
                       ' ' +
                       str(bbox[2]) +
                       ' ' +
                       str(bbox[3]) +
                       ' ' +
                       str(class_idx))
    txt_file.write('\n')
txt_file.close()