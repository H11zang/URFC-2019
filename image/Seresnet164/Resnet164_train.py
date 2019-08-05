# Copyright 2017 Queequeg92.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import print_function
#import pretrainedmodels
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets
from tensorboardX import SummaryWriter
#from torchvision import models as models2
from datetime import datetime
import numpy as np
import os
import sys
import time
import argparse
import models
from torch.autograd import Variable
from utils import mean_cifar10, std_cifar10, mean_cifar100, std_cifar100
from utils import AverageMeter

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR Classification Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet164',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: SeResNet164)')
print(model_names)
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=151, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')#0.025  hou 0.1
parser.add_argument('--lr_schedule', default=0, type=int,
                    help='learning rate schedule to apply')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=False, action='store_true', help='nesterov momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt_path', default='./logs/July06  09-04-16-0.54-40w/last_ckpt', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')

torch.cuda.set_device(1)

def main():
    nottrained=True;
    global args,correct_best
    args = parser.parse_args()
    # Data preprocessing.
    print('==> Preparing data......')
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100'), "Only support cifar10 or cifar100 dataset"
    if args.dataset == 'cifar10':
        print('To train and eval on cifar10 dataset......')
        num_classes = 9
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        transform_val = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])

        train_dir ='./fusai/train_data/'
        train_set = datasets.ImageFolder(train_dir,transform=transform_train)
        test_dir='./fusai/test_data/'
        test_set = datasets.ImageFolder(test_dir,transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
        val_dir ='./fusai/val_data1/'
        val_set = datasets.ImageFolder(val_dir,transform=transform_val)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)#shuffle=True 训练时打乱数据,设定了随机采样器，shuffle必须是Flase
        val_loader =torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4) 
    else:
        print('To train and eval on cifar100 dataset......')
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        transform_val = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)


    # Model
    if args.resume:
        # Load checkpoint.
        nottrained=False;
        print('==> Resuming from checkpoint..')
        model = models.__dict__[args.arch](num_classes)
        assert os.path.isdir(args.ckpt_path), 'Error: checkpoint directory not exists!'
        start_epoch = 0
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path,str(start_epoch)+'.pth')))

    else:
        print('==> Building model..')
        start_epoch = args.start_epoch
        model= models.__dict__[args.arch](num_classes)
# 10 -> 9
        #model.load_state_dict(torch.load('./logs/July05  11-36-49-val-0.53-180.pth/last_ckpt/180.pth'))#加载上次训练的模型
        #####冻结cnn，只训练fc
        '''
        frozen_layers=nn.ModuleList([model.conv1,model.stage1,model.stage2,model.bn])
        for params in frozen_layers.parameters():
            params.requires_grad = False
        '''
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    #labelweight=torch.Tensor([0.4184,0.53305,1.1207,2.9013,1.1464,0.72167,1.1437,1.5472,1.3921])#不平衡数据，权重交叉熵(没有效果)
    labelweight=torch.Tensor([1,1,1,1,1,1,1,1,1])
    # Use GPUs if available.
    if torch.cuda.is_available():
        model.cuda()
        labelweight=labelweight.cuda()
        #model = torch.nn.DataParallel(model, device_ids=range(3))#多GPU并行
        #cudnn.benchmark = True

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss(labelweight)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          nesterov=args.nesterov,
                          weight_decay=args.weight_decay)

    log_dir = 'logs/' + datetime.now().strftime('%B%d  %H-%M-%S')#冒号不能作目录
    train_writer = SummaryWriter(os.path.join(log_dir ,'train'))
    test_writer = SummaryWriter(os.path.join(log_dir ,'test'))

    # Save argparse commandline to a file.
    with open(os.path.join(log_dir, 'commandline_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
            
    best_acc = 0  # best test accuracy
    if(nottrained):
        for epoch in range(start_epoch, args.epochs):
            # Learning rate schedule.
            lr = adjust_learning_rate(optimizer, epoch + 1)
            train_writer.add_scalar('lr', lr, epoch)
            train(train_loader,val_loader,model, criterion, optimizer, train_writer, epoch)
      
            # Eval on test set.
            num_iter = (epoch + 1) * len(train_loader)

            # Save checkpoint.
            print('Saving Checkpoint......')
            state = {
                'model': model.state_dict() if torch.cuda.is_available() else model,
                'best_acc': best_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(os.path.join(log_dir, 'last_ckpt')):
                os.mkdir(os.path.join(log_dir, 'last_ckpt'))
            if(epoch%5==0):
                torch.save(model.state_dict(), os.path.join(log_dir, 'last_ckpt', str(epoch)+'.pth'))#每5次训练保存一次模型 
            eval(val_loader, model, criterion, test_writer, start_epoch, 12,nottrained)#跑完一次训练集，便进行一次验证集测试
    else:
        nottrained=False;        
        eval(test_loader, model, criterion, test_writer, start_epoch, 12,nottrained)#测试集       
    train_writer.close()
    test_writer.close()
def adjust_learning_rate(optimizer, epoch):
    if args.lr_schedule == 0:
        #lr= args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))#第一个可以考虑改成30,第二个改成  太多会导致过拟合
        lr= args.lr * ((0.1 ** int(epoch >= 30)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 90))* (0.1 ** int(epoch >= 120)))
    elif args.lr_schedule == 1:
        lr = args.lr * ((0.1 ** int(epoch >= 150)) * (0.1 ** int(epoch >= 225)))
    elif args.lr_schedule == 2:
        lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120)))
    else:
        raise Exception("Invalid learning rate schedule!")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Training
def train(train_loader,val_loader,model, criterion, optimizer, writer, epoch):
    print('\nEpoch: %d -> Training' % epoch)
    # Set to train mode.
    model.train()
    sample_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        num_iter = epoch * len(train_loader) + batch_idx
        # Add summary to train images.
        writer.add_image('image', vutils.make_grid(inputs[0:4], normalize=False, scale_each=True), num_iter)
        # Add summary to conv1 weights.

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        # Compute gradients and do back propagation.
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.item()*inputs.size(0), inputs.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        acces.update(correct, inputs.size(0))
        # measure elapsed time
        sample_time.update(time.time() - end, inputs.size(0))
        end = time.time()
        sys.stdout.write('Loss: %.4f | Acc: %.4f%% (%5d/%5d) \r' % (losses.avg, 100. * acces.avg, acces.numerator, acces.denominator))
        sys.stdout.flush()   
    writer.add_scalar('loss', losses.avg, epoch)
    writer.add_scalar('acc', acces.avg, epoch)
    print('Loss: %.4f | Acc: %.4f%% (%d/%d)' % (losses.avg, 100. * acces.avg, acces.numerator, acces.denominator))

# Evaluating
def eval(test_loader, model, criterion, writer, epoch, num_iter,nottrained):
    print('\nEpoch: %d -> Evaluating' % epoch)
    # Set to eval mode.
    model.eval()
    losses = AverageMeter()
    acces = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        output_value, predicted = torch.max(outputs.data, 1)#获取最大概率
        correct = predicted.eq(targets.data).cpu().sum()
        correct_best=correct
        if(correct>correct_best):
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_ckpt', str(epoch)+'.pth')) 
        acces.update(correct, inputs.size(0))
        sys.stdout.write('Acc: %.4f%% (%5d/%5d) \r' % (100. * acces.avg, acces.numerator, acces.denominator))
        sys.stdout.flush()
       
        if(nottrained==False):
            CategoryID=predicted.item()#提取值，item()
            if batch_idx<10:
                AreaID='00000'+str(batch_idx)
            elif batch_idx<100:
                AreaID='0000'+str(batch_idx)
            elif batch_idx<1000:
                AreaID='000'+str(batch_idx)
            elif batch_idx<10000:
                AreaID='00'+str(batch_idx)
            with open('./result_data.txt', "a", encoding="utf-8")as f:
                temp = AreaID + "\t" +'00'+str(CategoryID+1) + "\n"
                f.write(temp)     
    print('Acc: %.4f%% (%5d/%5d) \r' % (100. * acces.avg, acces.numerator, acces.denominator))    

if __name__ == '__main__':
    main()


