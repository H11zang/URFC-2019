from __future__ import print_function
import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
import csv
from utils import *
from multimodal import MultiModalDataset,MultiModalNet,CosineAnnealingLR
from tqdm import tqdm 
from config import config
from datetime import datetime
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score,accuracy_score
import torch.nn.functional as F

device = torch.device("cuda:1" )

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("/home/liangc/URFC-2019/image/logs/"):
    os.mkdir("/home/liangc/URFC-2019/image/logs/")

log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
log.write('mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    loss  f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------|\n')

def train(train_loader,model,criterion,optimizer,epoch,valid_metrics,best_results,start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()

    model.train()
    for i,(images,target) in enumerate(train_loader):
        images = images.to(device)
        indx_target=target.clone()
        target = torch.from_numpy(np.array(target)).long().to(device)
        # compute output
        output = model(images)
        loss = criterion(output,target)
        losses.update(loss.item(),images.size(0))
        f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
        acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
        f1.update(f1_batch,images.size(0))
        acc.update(acc_score,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s' % (\
                "train", i/len(train_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                valid_metrics[0], valid_metrics[1],valid_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))
        print(message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [acc.avg,losses.avg,f1.avg]

# 2. evaluate function
def evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start):
    # only meter loss and f1 score
    losses = AverageMeter()
    f1 = AverageMeter()
    acc= AverageMeter()
    # switch mode for evaluation
    model.to(device)
    model.eval()
    list_class = []
    csv_name = ['buliding_id', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    filename = "/home/liangc/URFC-2019/image/val1.csv"
    with open(filename) as f:
         reader = csv.reader(f)
         reader_list = list(reader)
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.to(device)
            indx_target=target.clone()
            target = torch.from_numpy(np.array(target)).long().to(device)
            output = model(images_var)
            list_class += [[reader_list[i + 1][0], float(output[0][0]), float(output[0][1]), float(output[0][2]),
                            float(output[0][3]), float(output[0][4]), float(output[0][5]), float(output[0][6]),
                            float(output[0][7]), float(output[0][8])]]
            loss = criterion(output,target)
            losses.update(loss.item(),images_var.size(0))
            f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
            acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))        
            f1.update(f1_batch,images.size(0))
            acc.update(acc_score,images.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    acc.avg,losses.avg,f1.avg,
                    train_metrics[0], train_metrics[1],train_metrics[2],
                    str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                    time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        log.write("\n")
        #log.write(message)
        #log.write("\n")
    #test = pd.DataFrame(columns=csv_name,data=list_class,index=None)
    #test.to_csv('/home/liangc/BaiDuBigData19-URFC-master/submit/val_prob.csv', encoding='gbk')
    return [acc.avg,losses.avg,f1.avg]

# 3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    sample_submission_df = pd.read_csv("/home/liangc/URFC-2019/image/test.csv")
    filename = "/home/liangc/URFC-2019/image/test.csv"
    with open(filename) as f:
         reader = csv.reader(f)
         reader_list = list(reader)
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.to(device)
    model.eval()
    submit_results = []
    list_class = []
    csv_name = ['buliding_id', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    for i,(input,filepath) in tqdm(enumerate(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.to(device)
            y_pred = model(image_var)
            label = F.softmax(y_pred).cpu().data.numpy()
            list_class += [[reader_list[i+1][0], float(label[0][0]), float(label[0][1]), float(label[0][2]), float(label[0][3]), float(label[0][4]),float(label[0][5]),float(label[0][6]), float(label[0][7]), float(label[0][8])]]
            labels.append(label==np.max(label))
            filenames.append(filepath)
    for row in np.concatenate(labels):
        subrow=np.argmax(row)
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('/home/liangc/URFC-2019/image/submit/%s_bestloss_submission.csv'%config.model_name, index=None)
    test = pd.DataFrame(columns=csv_name,data=list_class,index=None)
    test.to_csv('/home/liangc/URFC-2019/image/submit/test_prob.csv', encoding='gbk')

# 4. val1
def val1(test_loader,model,folds):
    sample_submission_df = pd.read_csv("/home/liangc/URFC-2019/image/val1.csv")
    filename = "/home/liangc/URFC-2019/image/val1.csv"
    with open(filename) as f:
         reader = csv.reader(f)
         reader_list = list(reader)
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.to(device)
    model.eval()
    submit_results = []
    list_class = []
    csv_name = ['buliding_id', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    for i,(input,filepath) in tqdm(enumerate(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.to(device)
            y_pred = model(image_var)
            label = F.softmax(y_pred).cpu().data.numpy()
            list_class += [[reader_list[i+1][0], float(label[0][0]), float(label[0][1]), float(label[0][2]), float(label[0][3]), float(label[0][4]),float(label[0][5]),float(label[0][6]), float(label[0][7]), float(label[0][8])]]
            labels.append(label==np.max(label))
            filenames.append(filepath)
    for row in np.concatenate(labels):
        subrow=np.argmax(row)
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('/home/liangc/URFC-2019/image/submit/%s_bestloss_val1_submission.csv'%config.model_name, index=None)
    test = pd.DataFrame(columns=csv_name,data=list_class,index=None)
    test.to_csv('/home/liangc/URFC-2019/image/submit/val1_prob.csv', encoding='gbk')

# 4. val2
def val2(test_loader,model,folds):
    sample_submission_df = pd.read_csv("/home/liangc/URFC-2019/image/val2.csv")
    filename = "/home/liangc/URFC-2019/image/val2.csv"
    with open(filename) as f:
         reader = csv.reader(f)
         reader_list = list(reader)
    #3.1 confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.to(device)
    model.eval()
    submit_results = []
    list_class = []
    csv_name = ['buliding_id', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    for i,(input,filepath) in tqdm(enumerate(test_loader)):
        #3.2 change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.to(device)
            y_pred = model(image_var)
            label = F.softmax(y_pred).cpu().data.numpy()
            list_class += [[reader_list[i+1][0], float(label[0][0]), float(label[0][1]), float(label[0][2]), float(label[0][3]), float(label[0][4]),float(label[0][5]),float(label[0][6]), float(label[0][7]), float(label[0][8])]]
            labels.append(label==np.max(label))
            filenames.append(filepath)
    for row in np.concatenate(labels):
        subrow=np.argmax(row)
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('/home/liangc/URFC-2019/image/submit/%s_bestloss_val2_submission.csv'%config.model_name, index=None)
    test = pd.DataFrame(columns=csv_name,data=list_class,index=None)
    test.to_csv('/home/liangc/URFC-2019/image/submit/val2_prob.csv', encoding='gbk')

# 5. main function
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("/home/liangc/URFC-2019/image/logs/"):
        os.mkdir("/home/liangc/URFC-2019/image/logs/")
    
    #4.2 get model
    model=MultiModalNet("se_resnext101_32x4d",0.5)

    #4.3 optim & criterion
    #optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    criterion=nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    best_acc=0
    best_loss = np.inf
    best_f1 = 0
    best_results = [0,np.inf,0]
    val_metrics = [0,np.inf,0]
    resume = False
    if resume:
        checkpoint = torch.load(r'/home/liangc/URFC-2019/image/checkpoints/best_models/seresnext101_dpn92_defrog_multimodal_fold_0_model_best_loss.pth.tar')
        best_acc = checkpoint['best_acc']
        best_loss = checkpoint['best_loss']
        best_f1 = checkpoint['best_f1']
        start_epoch = checkpoint['epoch']

    #if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    #all_files = pd.read_csv("/home/liangc/BaiDuBigData19-URFC-master/train.csv")
    test_files = pd.read_csv("/home/liangc/URFC-2019/image/test.csv")
    train_data_list = pd.read_csv("/home/liangc/URFC-2019/image/train_32w_clean.csv")
    val_data_list = pd.read_csv("/home/liangc/URFC-2019/image/val1.csv")
    val_data_list2 = pd.read_csv("/home/liangc/URFC-2019/image/val2.csv")
    #train_data_list,val_data_list = train_test_split(all_files, test_size=0.1, random_state = 2050)

    # load dataset
    train_gen = MultiModalDataset(train_data_list,config.train_data,mode="train")
    train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=False,num_workers=1) #num_worker is limited by shared memory in Docker!
    val_gen = MultiModalDataset(val_data_list,config.train_data,augument=False,mode="train")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,shuffle=False,pin_memory=False,num_workers=1)

    test_gen = MultiModalDataset(test_files,config.test_data,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=False,num_workers=1)
    valtest_gen = MultiModalDataset(val_data_list,config.train_data,augument=False,mode="test")
    valtest_loader = DataLoader(valtest_gen,1,shuffle=False,pin_memory=False,num_workers=1)
    valtest_gen2 = MultiModalDataset(val_data_list2,config.train_data,augument=False,mode="test")
    valtest_loader2 = DataLoader(valtest_gen2,1,shuffle=False,pin_memory=False,num_workers=1)
    #scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    #n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
    #scheduler = CosineAnnealingLR(optimizer, T_max=n_batches*2)
    start = timer()

    #train
    for epoch in range(0,config.epochs):
        scheduler.step(epoch)
        # train
        train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start)
        # val
        val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
        # check results
        is_best_acc=val_metrics[0] > best_results[0] 
        best_results[0] = max(val_metrics[0],best_results[0])
        is_best_loss = val_metrics[1] < best_results[1]
        best_results[1] = min(val_metrics[1],best_results[1])
        is_best_f1 = val_metrics[2] > best_results[2]
        best_results[2] = max(val_metrics[2],best_results[2])   
        # save model
        save_checkpoint({
                    "epoch":epoch + 1,
                    "model_name":config.model_name,
                    "state_dict":model.state_dict(),
                    "best_acc":best_results[0],
                    "best_loss":best_results[1],
                    "optimizer":optimizer.state_dict(),
                    "fold":fold,
                    "best_f1":best_results[2],
        },is_best_acc,is_best_loss,is_best_f1,fold)
        # print logs
        print('\r',end='',flush=True)
        log.write('%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % (\
                "best", epoch, epoch,                    
                train_metrics[0], train_metrics[1],train_metrics[2],
                val_metrics[0],val_metrics[1],val_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'))
            )
        log.write("\n")
        time.sleep(0.01)

    best_model = torch.load("%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold)
    val1(valtest_loader, model, fold)
    val2(valtest_loader2, model, fold)
if __name__ == "__main__":
    main()
