#!/usr/bin/env sh
snapshot_dir="./snapshot"
if [ ! -d "$snapshot_dir" ]; then
	mkdir "$snapshot_dir"
fi

# yolo2caffe="src/yolo2caffe.py"
# train_yolo="src/yolo_train.prototxt"
# newer=`find $yolo2caffe -newer $train_yolo`
# # if .prototxt is older
# if [ ! -d "$train_yolo" -o "$train_yolo" -ot "$yolo2caffe"]; then
# 	echo "create $train_yolo"
# 	python "$yolo2caffe"
# 	# if [ $?==1 ]; then
# 	# 	exit
# 	# fi
# fi
./caffe/build/tools/caffe train --solver=src/yolo_solver.prototxt
