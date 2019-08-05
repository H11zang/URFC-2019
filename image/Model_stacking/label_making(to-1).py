import csv
import pandas as pd
import time
import numpy as np
import sys
import datetime
import os

#历遍文件夹
def findtxt(path, ret):
    print('开始遍历文件夹')
    n = 0
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".jpg"):
                ret.append(de_path)
        else:
            findtxt(de_path, ret)

#输出csv文件
def write_csv(csv_N,list_w):
    csv_name = ['Id','Target']
    test = pd.DataFrame(columns=csv_name, data=list_w)
    test.to_csv(csv_N, encoding='gbk', index=None)

#输出训练集目录
def label_make_train(path):
    ret = []
    list_w = []
    csv_N = 'train_40w.csv'
    findtxt(path, ret)
    length = len(ret)
    start_time = time.time()
    for index, filename in enumerate(ret):
        id = str(filename[filename.rfind(".")-10:filename.rfind(".")])
        label = str(filename[filename.rfind(".")-3:filename.rfind(".")])
        label = str(int(label)-1)
        list_w += [[id,label]]
        sys.stdout.write('\r>> Processing visit data %d/%d'%(index+1, length))
        sys.stdout.flush()
    write_csv(csv_N, list_w)
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))

#输出测试集目录
def label_make_test(path):
    ret = []
    list_w = []
    csv_N = 'test.csv'
    findtxt(path, ret)
    length = len(ret)
    start_time = time.time()
    for index, filename in enumerate(ret):
        id = str(filename[filename.rfind(".")-6:filename.rfind(".")])
        label = 0
        list_w += [[id,label]]
        sys.stdout.write('\r>> Processing visit data %d/%d'%(index+1, length))
        sys.stdout.flush()
    write_csv(csv_N, list_w)
    sys.stdout.write('\n')
    print("using time:%.2fs"%(time.time()-start_time))

if __name__ == '__main__':
    label_make_train('/home/liangc/URFC-2019/visit+image（baseline）/data/train/')
    label_make_test('/home/liangc/URFC-2019/visit+image（baseline）/data/test/')