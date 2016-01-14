import os

output = '''name: "YoloNet"
layer {
	name: "yolo"
	type: "WindowData"
	top: "data"
	top: "label"
	include {
		phase: TRAIN
	}
	transform_param {
		mirror: true
		scale: 0.00390625
		crop_size: 448
	}
	window_data_param {
		source: "data/2012_train_caffe.txt"
		batch_size: 64
		crop_size: 448
		fg_threshold: 0.5
		bg_threshold: 0.5
	}
}
layer {
	name: "yolo"
	type: "WindowData"
	top: "data"
	top: "label"
	include {
		phase: TEST
	}
	transform_param {
		mirror: true
		scale: 0.00390625
		crop_size: 448
	}
	window_data_param {
		source: "data/2012_test_caffe.txt"
		batch_size: 64
		crop_size: 448
		fg_threshold: 0.5
		bg_threshold: 0.5
	}
}
'''

conv_template = '''
layer {
	name: "%(name)s"
	type: "Convolution"
	bottom: "%(bottom)s"
	top: "%(top)s"
	convolution_param {
		num_output: %(filters)s
		pad: %(pad)s
		kernel_size: %(kernel)s
		stride: %(stride)s
		weight_filler {
			type: "xavier"
		}
	}
}
layer {
	name: "%(leaky_name)s"
	type: "ReLU"
	bottom: "%(top)s"
	top: "%(top)s"
	relu_param {
		negative_slope: 0.1
	}
}
'''

maxpool_template = '''
layer {
	name: "%(name)s"
	type: "Pooling"
	bottom: "%(bottom)s"
	top: "%(top)s"
	pooling_param {
		pool: MAX
		kernel_size: %(kernel)s
		stride: %(stride)s
	}
}
'''

with open("/home/yushroom/program/github/darknet/cfg/yolo.cfg") as f:
	yolo_cfg_file = f.read()
yolo_cfg = yolo_cfg_file.split('\n')

layer_name_white_list = ('[convolutional]', '[maxpool]', '[connected]', '[dropout]')
cfg_idx = 0
conv_group = 0
maxpool_group = 0
last_layer_name = 'data'

while cfg_idx < len(yolo_cfg):
	line = yolo_cfg[cfg_idx]

	# conv
	if line == '[convolutional]':
		# print "conv"
		filters = yolo_cfg[cfg_idx+1].split('=')[-1]
		size 	= yolo_cfg[cfg_idx+2].split('=')[-1]
		stride	= yolo_cfg[cfg_idx+3].split('=')[-1]
		pad		= yolo_cfg[cfg_idx+4].split('=')[-1]
		#print conv_template
		value_map = {}
		value_map['name'] 		= 'conv{0}'.format(conv_group)
		value_map['bottom'] 	= last_layer_name
		value_map['top'] 		= value_map['name']
		value_map['filters'] 	= filters
		value_map['pad'] 		= pad
		value_map['kernel'] 	= size
		value_map['stride'] 	= stride
		value_map['leaky_name'] = value_map['name']+'leaky'
		output += conv_template % value_map
		cfg_idx += 6
		last_layer_name = value_map['name']
		conv_group += 1
	
	elif line == 'maxpool':
		size 	= yolo_cfg[cfg_idx+2].split('=')[-1]
		stride	= yolo_cfg[cfg_idx+3].split('=')[-1]
		value_map = {}
		value_map['name'] 		= 'pool{0}'.format(maxpool_group)
		value_map['bottom'] 	= last_layer_name
		value_map['top'] 		= value_map['name']
		value_map['kernel'] 	= size
		value_map['stride'] 	= stride
		output += maxpool_template % value_map
		cfg_idx += 3
		maxpool_group += 1

	else:
		cfg_idx += 1

output += '''
layer {
	name: "fc1"
	type: "InnerProduct"
	bottom: "%(bottom)s"
	top: "fc1"
	inner_product_param {
		num_output: 4096
	}
}
layer {
	name: "fc1leaky"
	type: "Leaky"
	bottom: "fc1"
	top: "fc1"
}
layer {
	name: "drop_fc"
	type: "Dropout"
	bottom: "fc1"
	top: "fc1"
	dropout_param {
		dropout_ratio: 0.5
	}
}
layer {
	name: "fc2"
	type: "InnerProduct"
	bottom: "fc1"
	top: "fc2"
	inner_product_param {
		num_output: 1470
	}
}
''' % {"bottom": last_layer_name}

with open("yolo_train.prototxt", "w") as f:
	f.write(output)