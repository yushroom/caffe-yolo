import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2012', 'train'), ('2012', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def get_annotation(year, image_id):
    ret_data = []
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    ret_data.append(3)
    ret_data.append(w)
    ret_data.append(h)
    ret_data.append(0)

    bbox_count = 0;
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (classes.index(cls)+1, int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
        #bb = convert((w,h), b)
        #out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        ret_data.append(b)
        bbox_count += 1
    ret_data[3]=bbox_count
    return ret_data

wd = getcwd()

def bbox_data_2_strings(bbox_data):
    strs = []
    for i in range(0, 4):
        strs.append(str(bbox_data[i]))
    #strs.append(str(bbox_data[3]))
    for i in range(0, bbox_data[3]):
        strs.append("%d %d %d %d %d" % bbox_data[i+4])
    return strs

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    #list_file = open('%s_%s.txt'%(year, image_set), 'w')
    list_file = open('%s_%s_caffe.txt'%(year, image_set), 'w')
    list_file.write("%d\n" % (len(image_ids)))
    image_index = 0
    for image_id in image_ids:
        list_file.write('# %d\n' % (image_index))
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        #convert_annotation(year, image_id)
        bbox_data = get_annotation(year, image_id)
        list_file.write('\n'.join(bbox_data_2_strings(bbox_data)))
        list_file.write('\n')
        image_index += 1
    list_file.close()

