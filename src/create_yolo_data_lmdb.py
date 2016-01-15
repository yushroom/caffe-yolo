import caffe
import lmdb
from PIL import Image
import numpy as np
import os
import shutil

target_size = (448, 448)

with open("../data/2012_train_caffe.txt") as f:
	lists = f.read().split('\n')
total = int(lists[0])
idx = 1
images = []
label_and_bbox_s = []
while idx < len(lists):
#while idx < 10:
	if lists[idx].startswith("#"):
		images.append(lists[idx+1])
		channel = int(lists[idx+2])
		width   = int(lists[idx+3])
		height  = int(lists[idx+4])
		num_box = int(lists[idx+5])
		lb = np.zeros((1, 64, 5))
		for i in range(0, num_box):
			box = [ float(s) for s in lists[idx+5+1+i].split(' ')]
			box[1] /= width
			box[3] /= width
			box[2] /= height
			box[4] /= height
			lb[0, i] = np.array(box)
		label_and_bbox_s.append(lb)
		idx += 5+num_box
	idx+=1

if False:
	images = images[0:10]
	label_and_bbox_s = label_and_bbox_s[0:10]
	print label_and_bbox_s[0][0]

#print images
print "train set: %d images" % (len(images))

data_lmdb = 'yolo-train-lmdb'
label_lmdb = 'yolo-train-label-lmdb'

if os.path.exists(data_lmdb):
	shutil.rmtree(data_lmdb)
if os.path.exists(label_lmdb):
	shutil.rmtree(label_lmdb)

print "write image data to lmdb..."
image_db = lmdb.open(data_lmdb, map_size=int(1e12))
with image_db.begin(write=True) as txn:
	for idx, image_fn in enumerate(images):
		if idx % 500 == 0:
			print idx
		# load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
		im = np.array(Image.open(image_fn).resize((448, 448), Image.BILINEAR))
		im = im[:, :, ::-1]
		im = im.transpose((2, 0, 1))
		im_dat = caffe.io.array_to_datum(im)
		txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())
image_db.close()
print "done."

print "write image label to lmdb..."
label_db = lmdb.open(label_lmdb, map_size=int(1e12))
with label_db.begin(write=True) as txn:
	for idx, lb in enumerate(label_and_bbox_s):
		if idx % 500 == 0:
			print idx
		lb_data = caffe.io.array_to_datum(lb)
		txn.put('{:0>10d}'.format(idx), lb_data.SerializeToString())
label_db.close()
print "done."