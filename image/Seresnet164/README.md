Environmental Requirement:
Python 3.7
Pytorch 1.0.1
torchvision 0.2.2
numpy
tensorboardX

使用说明：
(1)采用resnet164模型训练，运行
python Resnet164_train.py --start_epoch 0 --batch_size 128
(2)采用resnet164模型获取9种概率，需要先修改models目录下的resnet.py的注释为9种概率的forward方法，再运行
python Resnet164_test_9proba.py --start_epoch 0 --batch_size 1
(3)采用resnet164模型获取256channel，需要先修改models目录下的resnet.py的注释为256channel的forward方法，再运行
python Resnet164_test_256channel.py --start_epoch 0 --batch_size 1
