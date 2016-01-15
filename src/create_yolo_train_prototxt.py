import caffe
import lmdb
from caffe import layers as L
from caffe import params as P

def ConvLayer(bottom, filters, size, stride, pad):
    return L.Convolution(bottom, num_output=filters, kernel_size=size, stride=stride, pad=pad, weight_filler=dict(type='xavier'))

def LeakyLayer(bottom):
    return L.ReLU(bottom, relu_param=dict(negative_slope=0.1), in_place=True)

def MaxpoolingLayer(bottom, size, stride):
    return L.Pooling(bottom, kernel_size=size, stride=2, pool=P.Pooling.MAX)

def yolo_net(data_lmdb, label_lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    # input
    n.data = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=data_lmdb, transform_param=dict(scale=1./255), ntop=1)
    n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=label_lmdb, ntop=1)

    # 7x7x64-s-2
    n.conv1  = ConvLayer(n.data, 64, 7, 2, 1)
    n.leaky1 = LeakyLayer(n.conv1)
    n.pool1  = MaxpoolingLayer(n.leaky1, 2, 2)

    # 3x3x192
    n.conv2  = ConvLayer(n.pool1, 192, 3, 1, 1)
    n.leaky2 = LeakyLayer(n.conv2)
    n.pool2  = MaxpoolingLayer(n.leaky2, 2, 2)

    n.conv3  = ConvLayer(n.pool2, 128, 1, 1, 1)
    n.leaky3 = LeakyLayer(n.conv3)
    n.conv4  = ConvLayer(n.leaky3, 256, 3, 1, 1)
    n.leaky4 = LeakyLayer(n.conv4)
    n.conv5  = ConvLayer(n.leaky4, 256, 1, 1, 1)
    n.leaky5 = LeakyLayer(n.conv5)
    n.conv6  = ConvLayer(n.leaky5, 512, 3, 1, 1)
    n.leaky6 = LeakyLayer(n.conv6)
    n.pool3  = MaxpoolingLayer(n.leaky6, 2, 2)

    n.conv7  = ConvLayer(n.pool3, 256, 1, 1, 1)
    n.leaky7 = LeakyLayer(n.conv7)
    n.conv8  = ConvLayer(n.leaky7, 512, 3, 1, 1)
    n.leaky8 = LeakyLayer(n.conv8)
    n.conv9  = ConvLayer(n.leaky8, 256, 1, 1, 1)
    n.leaky9 = LeakyLayer(n.conv9)
    n.conv10  = ConvLayer(n.leaky9, 512, 3, 1, 1)
    n.leaky10 = LeakyLayer(n.conv10)
    n.conv11  = ConvLayer(n.leaky10, 256, 1, 1, 1)
    n.leaky11 = LeakyLayer(n.conv11)
    n.conv12  = ConvLayer(n.leaky11, 512, 3, 1, 1)
    n.leaky12 = LeakyLayer(n.conv12)
    n.conv13  = ConvLayer(n.leaky12, 256, 1, 1, 1)
    n.leaky13 = LeakyLayer(n.conv13)
    n.conv14  = ConvLayer(n.leaky13, 512, 3, 1, 1)
    n.leaky14 = LeakyLayer(n.conv14)
    n.conv15  = ConvLayer(n.leaky14, 512, 1, 1, 1)
    n.leaky15 = LeakyLayer(n.conv15)
    n.conv16  = ConvLayer(n.leaky15, 1024, 3, 1, 1)
    n.leaky16 = LeakyLayer(n.conv16)
    n.pool4  = MaxpoolingLayer(n.leaky16, 2, 2)

    n.conv17  = ConvLayer(n.pool4, 512, 1, 1, 1)
    n.leaky17 = LeakyLayer(n.conv17)
    n.conv18  = ConvLayer(n.leaky17, 1024, 3, 1, 1)
    n.leaky18 = LeakyLayer(n.conv18)
    n.conv19  = ConvLayer(n.leaky18, 512, 1, 1, 1)
    n.leaky19 = LeakyLayer(n.conv19)
    n.conv20  = ConvLayer(n.leaky19, 1024, 3, 1, 1)
    n.leaky20 = LeakyLayer(n.conv20)
    n.pool5  = MaxpoolingLayer(n.leaky20, 2, 2)

    n.conv21  = ConvLayer(n.pool5, 512, 1, 1, 1)
    n.leaky21 = LeakyLayer(n.conv21)
    n.conv22  = ConvLayer(n.leaky21, 1024, 3, 1, 1)
    n.leaky22 = LeakyLayer(n.conv22)
    n.conv23  = ConvLayer(n.leaky22, 512, 1, 1, 1)
    n.leaky23 = LeakyLayer(n.conv23)
    n.conv24  = ConvLayer(n.leaky23, 1024, 3, 1, 1)
    n.leaky24 = LeakyLayer(n.conv24)

    n.fc1 = L.InnerProduct(n.leaky24, num_output=4096, weight_filler=dict(type='xavier'))
    n.leaky25 = LeakyLayer(n.fc1)
    n.dropout = L.Dropout(n.leaky25, dropout_ratio=0.5, in_place=True)
    n.fc2 = L.InnerProduct(n.dropout, num_output=1470, weight_filler=dict(type='xavier'))

    return n.to_proto()
    
with open('yolo_auto_train.prototxt', 'w') as f:
    f.write(str(yolo_net('src/yolo-train-lmdb', 'src/yolo-train-label-lmdb', 64)))