import caffe
import lmdb
from PIL import Image
import numpy as np

with open("../data/2012_train.txt") as f:
	images = f.read().split('\n')
images = images[0:10]
print "train set: %d images" % (len(images))

image_db = lmdb.open('yolo-train-lmdb', map_size=int(1e12))
with image_db.begin(write=True) as txn:
	for idx, image_fn in enumerate(images):
		im = np.array(Image.open(image_fn))
		im = im[:, :, ::-1]
		im = im.transpose((2, 0, 1))
		im_dat = caffe.io.array_to_datum(im)
		txn.put('{:0>10d}'.format(idx), im_dat.SerializeToString())
image_db.close()