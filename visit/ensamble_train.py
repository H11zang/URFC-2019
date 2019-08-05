import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv

#预测类别
def y_pred_max(num):
    list_pred = []
    for i in range(len(num)):
        max = 0
        max_id = 0
        for j in range(9):
            if float(num[i][j]) > max:
                max =float(num[i][j])
                max_id = j
        list_pred +=[max_id]
    return list_pred

#线下检测类别及精确度
def class_val():
    max_id_list = ["009", "001", "002", "003", "004", "005", "006", "007", "008", "009"]
    userFile_class = "class_num.txt"
    class_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_num_sum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_num_p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_num_10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_num_sum_10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_num_p_10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num = len(pred_max)
    num_id = 0
    sum_ture = 0
    sum = 0
    while (num):
        if int(pred_max[num_id]) == int(reader_val[num_id+1][1]):
            class_num[int(pred_max[num_id])] += 1
            if int(reader_val[num_id+1][3]) <=10:
                class_num_10[int(pred_max[num_id])] += 1
        if int(reader_val[num_id + 1][3]) <= 10:
            class_num_sum_10[int(reader_val[num_id + 1][1])] += 1
            class_num_p_10[int(pred_max[num_id])] += 1
        class_num_p[int(pred_max[num_id])] += 1
        class_num_sum[int(reader_val[num_id+1][1])] += 1
        num -= 1
        num_id += 1
    class_num[9] = class_num[0]
    class_num_sum[9] = class_num_sum[0]
    class_num_p[9] = class_num_p[0]
    class_num_10[9] = class_num_10[0]
    class_num_sum_10[9] = class_num_sum_10[0]
    class_num_p_10[9] = class_num_p_10[0]
    num = len(class_num) - 1
    num_id = 1
    with open(userFile_class, "a", encoding="utf-8")as f:
        temp = "\n"
        f.write(temp)
        temp = 'class' + "\t" + 'True' + "\t" + 'Sum' + "\t" + 'Prob_sum' + "\t" + 'class_score' + "\t" + "\n"
        f.write(temp)
        while (num):
            temp = max_id_list[num_id] + "\t" + str(class_num[num_id]) + "\t" + str(class_num_sum[num_id]) + "\t" + str(class_num_p[num_id]) + "\t" + str(round(class_num[num_id]/class_num_sum[num_id],3)) + "\n"
            sum_ture += class_num[num_id]
            sum += class_num_sum[num_id]
            f.write(temp)
            num_id += 1
            num -= 1
        temp = 'score' + "\t" + str(round(sum_ture / sum, 3)) + "\n"
        f.write(temp)

#输出csv文件
def write_csv(name_csv,state):
    list_class = []
    if state == 'val':
        csv_name = ['buliding_id', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8', 'ST9', 'predicted', 'label']
        for i in range(len(y_pred)):
            list_class += [
                [reader_val[i + 1][0], float(y_pred[i][1]), float(y_pred[i][2]), float(y_pred[i][3]),
                 float(y_pred[i][4]), float(y_pred[i][5]),
                 float(y_pred[i][6]), float(y_pred[i][7]), float(y_pred[i][8]), float(y_pred[i][0]), pred_max[i],
                 reader_val[i + 1][1]]]
        test = pd.DataFrame(columns=csv_name, data=list_class)
        test.to_csv(name_csv, encoding='gbk', index=None)
    if state == 'test':
        csv_name = ['buliding_id', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8', 'ST9', 'predicted']
        for i in range(len(y_pred)):
            list_class += [
                [reader_test[i + 1][0], float(y_pred[i][1]), float(y_pred[i][2]), float(y_pred[i][3]),
                 float(y_pred[i][4]), float(y_pred[i][5]),
                 float(y_pred[i][6]), float(y_pred[i][7]), float(y_pred[i][8]), float(y_pred[i][0]), pred_max[i]]]
        test = pd.DataFrame(columns=csv_name, data=list_class)
        test.to_csv(name_csv, encoding='gbk', index=None)

url_val = 'dataset/dataset_val2.csv'
with open(url_val) as f:
    reader = csv.reader(f)
    reader_val = list(reader)

url_test = 'dataset/dataset_test.csv'
with open(url_test) as f:
    reader = csv.reader(f)
    reader_test = list(reader)

######导入val1、val2######
##level1
dataset1 = pd.read_csv('dataset/dataset_val1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset1_val = pd.read_csv('dataset/dataset_val2.csv')
dataset1_val.label.replace(-1,0,inplace=True)
dataset1.drop_duplicates(inplace=True)
dataset1_val.drop_duplicates(inplace=True)
dataset1_x = dataset1.drop(['buliding_id','label'],axis=1)
dataset1_val_x = dataset1_val.drop(['buliding_id','label'],axis=1)

##level1(image256)
dataset1_image256 = pd.read_csv('dataset/dataset_val1(image_256_32wmodel).csv')
dataset1_image256.label.replace(-1,0,inplace=True)
dataset1_val_image256 = pd.read_csv('dataset/dataset_val2(image_256_32wmodel).csv')
dataset1_val_image256.label.replace(-1,0,inplace=True)
dataset1_image256.drop_duplicates(inplace=True)
dataset1_val_image256.drop_duplicates(inplace=True)
dataset1_x_image256 = dataset1_image256.drop(['buliding_id','label'],axis=1)
dataset1_val_x_image256 = dataset1_val_image256.drop(['buliding_id','label'],axis=1)

##Visit--xgbooost
dataset_xgb = pd.read_csv('dataset/visit_xgb/val1_8.csv')
dataset_xgb.label.replace(-1,0,inplace=True)
dataset_val_xgb = pd.read_csv('dataset/visit_xgb/val2_8.csv')
dataset_val_xgb.label.replace(-1,0,inplace=True)
dataset_xgb.drop_duplicates(inplace=True)
dataset_val_xgb.drop_duplicates(inplace=True)
dataset_x_xgb = dataset_xgb.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_xgb = dataset_val_xgb.drop(['buliding_id','label','predicted'],axis=1)

##F1_1
datasetf1_1 = pd.read_csv('dataset/visit_xgb/val1_1_1.csv')
datasetf1_1.label.replace(-1,0,inplace=True)
datasetf1_1_val = pd.read_csv('dataset/visit_xgb/val2_1_1.csv')
datasetf1_1_val.label.replace(-1,0,inplace=True)
datasetf1_1.drop_duplicates(inplace=True)
datasetf1_1_val.drop_duplicates(inplace=True)
datasetf1_1_x = datasetf1_1.drop(['buliding_id','label','predicted'],axis=1)
datasetf1_1_val_x = datasetf1_1_val.drop(['buliding_id','label','predicted'],axis=1)

##F1_2
datasetf1_2 = pd.read_csv('dataset/visit_xgb/val1_1_2.csv')
datasetf1_2.label.replace(-1,0,inplace=True)
datasetf1_2_val = pd.read_csv('dataset/visit_xgb/val2_1_2.csv')
datasetf1_2_val.label.replace(-1,0,inplace=True)
datasetf1_2.drop_duplicates(inplace=True)
datasetf1_2_val.drop_duplicates(inplace=True)
datasetf1_2_x = datasetf1_2.drop(['buliding_id','label','predicted'],axis=1)
datasetf1_2_val_x = datasetf1_2_val.drop(['buliding_id','label','predicted'],axis=1)

##F1_3
datasetf1_3 = pd.read_csv('dataset/visit_xgb/val1_1_3.csv')
datasetf1_3.label.replace(-1,0,inplace=True)
datasetf1_3_val = pd.read_csv('dataset/visit_xgb/val2_1_3.csv')
datasetf1_3_val.label.replace(-1,0,inplace=True)
datasetf1_3.drop_duplicates(inplace=True)
datasetf1_3_val.drop_duplicates(inplace=True)
datasetf1_3_x = datasetf1_3.drop(['buliding_id','label','predicted'],axis=1)
datasetf1_3_val_x = datasetf1_3_val.drop(['buliding_id','label','predicted'],axis=1)

##F1_8
datasetf1_8 = pd.read_csv('dataset/visit_xgb/val1_1_8.csv')
datasetf1_8.label.replace(-1,0,inplace=True)
datasetf1_8_val = pd.read_csv('dataset/visit_xgb/val2_1_8.csv')
datasetf1_8_val.label.replace(-1,0,inplace=True)
datasetf1_8.drop_duplicates(inplace=True)
datasetf1_8_val.drop_duplicates(inplace=True)
datasetf1_8_x = datasetf1_8.drop(['buliding_id','label','predicted'],axis=1)
datasetf1_8_val_x = datasetf1_8_val.drop(['buliding_id','label','predicted'],axis=1)

##F7_1
datasetf7_1 = pd.read_csv('dataset/visit_xgb/val1_7_1.csv')
datasetf7_1.label.replace(-1,0,inplace=True)
datasetf7_1_val = pd.read_csv('dataset/visit_xgb/val2_7_1.csv')
datasetf7_1_val.label.replace(-1,0,inplace=True)
datasetf7_1.drop_duplicates(inplace=True)
datasetf7_1_val.drop_duplicates(inplace=True)
datasetf7_1_x = datasetf7_1.drop(['buliding_id','label','predicted'],axis=1)
datasetf7_1_val_x = datasetf7_1_val.drop(['buliding_id','label','predicted'],axis=1)

##F7_2
datasetf7_2 = pd.read_csv('dataset/visit_xgb/val1_7_2.csv')
datasetf7_2.label.replace(-1,0,inplace=True)
datasetf7_2_val = pd.read_csv('dataset/visit_xgb/val2_7_2.csv')
datasetf7_2_val.label.replace(-1,0,inplace=True)
datasetf7_2.drop_duplicates(inplace=True)
datasetf7_2_val.drop_duplicates(inplace=True)
datasetf7_2_x = datasetf7_2.drop(['buliding_id','label','predicted'],axis=1)
datasetf7_2_val_x = datasetf7_2_val.drop(['buliding_id','label','predicted'],axis=1)

##F7_3
datasetf7_3 = pd.read_csv('dataset/visit_xgb/val1_7_3.csv')
datasetf7_3.label.replace(-1,0,inplace=True)
datasetf7_3_val = pd.read_csv('dataset/visit_xgb/val2_7_3.csv')
datasetf7_3_val.label.replace(-1,0,inplace=True)
datasetf7_3.drop_duplicates(inplace=True)
datasetf7_3_val.drop_duplicates(inplace=True)
datasetf7_3_x = datasetf7_3.drop(['buliding_id','label','predicted'],axis=1)
datasetf7_3_val_x = datasetf7_3_val.drop(['buliding_id','label','predicted'],axis=1)

##F7_5
datasetf7_5 = pd.read_csv('dataset/visit_xgb/val1_7_5.csv')
datasetf7_5.label.replace(-1,0,inplace=True)
datasetf7_5_val = pd.read_csv('dataset/visit_xgb/val2_7_5.csv')
datasetf7_5_val.label.replace(-1,0,inplace=True)
datasetf7_5.drop_duplicates(inplace=True)
datasetf7_5_val.drop_duplicates(inplace=True)
datasetf7_5_x = datasetf7_5.drop(['buliding_id','label','predicted'],axis=1)
datasetf7_5_val_x = datasetf7_5_val.drop(['buliding_id','label','predicted'],axis=1)

##F7_7
datasetf7_7 = pd.read_csv('dataset/visit_xgb/val1_7_7.csv')
datasetf7_7.label.replace(-1,0,inplace=True)
datasetf7_7_val = pd.read_csv('dataset/visit_xgb/val2_7_7.csv')
datasetf7_7_val.label.replace(-1,0,inplace=True)
datasetf7_7.drop_duplicates(inplace=True)
datasetf7_7_val.drop_duplicates(inplace=True)
datasetf7_7_x = datasetf7_7.drop(['buliding_id','label','predicted'],axis=1)
datasetf7_7_val_x = datasetf7_7_val.drop(['buliding_id','label','predicted'],axis=1)

##Fadd1
datasetfadd1 = pd.read_csv('dataset/visit_xgb/val1_add1.csv')
datasetfadd1.label.replace(-1,0,inplace=True)
datasetfadd1_val = pd.read_csv('dataset/visit_xgb/val2_add1.csv')
datasetfadd1_val.label.replace(-1,0,inplace=True)
datasetfadd1.drop_duplicates(inplace=True)
datasetfadd1_val.drop_duplicates(inplace=True)
datasetfadd1_x = datasetfadd1.drop(['buliding_id','label','predicted'],axis=1)
datasetfadd1_val_x = datasetfadd1_val.drop(['buliding_id','label','predicted'],axis=1)

##Fadd2
datasetfadd2 = pd.read_csv('dataset/visit_xgb/val1_add2.csv')
datasetfadd2.label.replace(-1,0,inplace=True)
datasetfadd2_val = pd.read_csv('dataset/visit_xgb/val2_add2.csv')
datasetfadd2_val.label.replace(-1,0,inplace=True)
datasetfadd2.drop_duplicates(inplace=True)
datasetfadd2_val.drop_duplicates(inplace=True)
datasetfadd2_x = datasetfadd2.drop(['buliding_id','label','predicted'],axis=1)
datasetfadd2_val_x = datasetfadd2_val.drop(['buliding_id','label','predicted'],axis=1)

##图像
dataset_m = pd.read_csv('dataset/class_dataset/m_val1.csv')
dataset_m.label.replace(-1,0,inplace=True)
dataset_val_m = pd.read_csv('dataset/class_dataset/m_val2.csv')
dataset_val_m.label.replace(-1,0,inplace=True)
dataset_m.drop_duplicates(inplace=True)
dataset_val_m.drop_duplicates(inplace=True)
dataset_x_m = dataset_m.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_m = dataset_val_m.drop(['buliding_id','label','predicted'],axis=1)

##图像(数据增强版)
dataset_ms = pd.read_csv('dataset/class_dataset/ms_val1.csv')
dataset_ms.label.replace(-1,0,inplace=True)
dataset_val_ms = pd.read_csv('dataset/class_dataset/ms_val2.csv')
dataset_val_ms.label.replace(-1,0,inplace=True)
dataset_ms.drop_duplicates(inplace=True)
dataset_val_ms.drop_duplicates(inplace=True)
dataset_x_ms = dataset_ms.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_ms = dataset_val_ms.drop(['buliding_id','label','predicted'],axis=1)

##image256--xgboost
dataset_xgb_image256 = pd.read_csv('dataset/class_dataset/x_val1(image_256).csv')
dataset_xgb_image256.label.replace(-1,0,inplace=True)
dataset_val_xgb_image256 = pd.read_csv('dataset/class_dataset/x_val2(image_256).csv')
dataset_val_xgb_image256.label.replace(-1,0,inplace=True)
dataset_xgb_image256.drop_duplicates(inplace=True)
dataset_val_xgb_image256.drop_duplicates(inplace=True)
dataset_x_xgb_image256 = dataset_xgb_image256.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_xgb_image256 = dataset_val_xgb_image256.drop(['buliding_id','label','predicted'],axis=1)

##图像(efficientnet_fc)
dataset_EF = pd.read_csv('dataset/class_dataset/EF_val1.csv')
dataset_EF.label.replace(-1,0,inplace=True)
dataset_val_EF = pd.read_csv('dataset/class_dataset/EF_val2.csv')
dataset_val_EF.label.replace(-1,0,inplace=True)
dataset_EF.drop_duplicates(inplace=True)
dataset_val_EF.drop_duplicates(inplace=True)
dataset_x_EF = dataset_EF.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_EF = dataset_val_EF.drop(['buliding_id','label','predicted'],axis=1)

##图像(se_ef)
dataset_se_ef = pd.read_csv('dataset/class_dataset/se_ef_val1.csv')
dataset_se_ef.label.replace(-1,0,inplace=True)
dataset_val_se_ef = pd.read_csv('dataset/class_dataset/se_ef_val2.csv')
dataset_val_se_ef.label.replace(-1,0,inplace=True)
dataset_se_ef.drop_duplicates(inplace=True)
dataset_val_se_ef.drop_duplicates(inplace=True)
dataset_x_se_ef = dataset_se_ef.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_se_ef = dataset_val_se_ef.drop(['buliding_id','label','predicted'],axis=1)

##图像(se2)
dataset_se2 = pd.read_csv('dataset/class_dataset/se2_val1.csv')
dataset_se2.label.replace(-1,0,inplace=True)
dataset_val_se2 = pd.read_csv('dataset/class_dataset/se2_val2.csv')
dataset_val_se2.label.replace(-1,0,inplace=True)
dataset_se2.drop_duplicates(inplace=True)
dataset_val_se2.drop_duplicates(inplace=True)
dataset_x_se2 = dataset_se2.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_se2 = dataset_val_se2.drop(['buliding_id','label','predicted'],axis=1)

##图像(se_ir)
dataset_se_ir = pd.read_csv('dataset/class_dataset/se_ir_val1.csv')
dataset_se_ir.label.replace(-1,0,inplace=True)
dataset_val_se_ir = pd.read_csv('dataset/class_dataset/se_ir_val2.csv')
dataset_val_se_ir.label.replace(-1,0,inplace=True)
dataset_se_ir.drop_duplicates(inplace=True)
dataset_val_se_ir.drop_duplicates(inplace=True)
dataset_x_se_ir = dataset_se_ir.drop(['buliding_id','label','predicted'],axis=1)
dataset_val_x_se_ir = dataset_val_se_ir.drop(['buliding_id','label','predicted'],axis=1)

datasetf1_x = pd.concat([datasetf1_1_x,datasetf1_2_x,datasetf1_3_x,datasetf1_8_x], axis=1)
datasetf1_val_x = pd.concat([datasetf1_1_val_x,datasetf1_2_val_x,datasetf1_3_val_x,datasetf1_8_val_x], axis=1)
datasetf7_x = pd.concat([datasetf7_1_x,datasetf7_2_x,datasetf7_3_x,datasetf7_5_x,datasetf7_7_x], axis=1)
datasetf7_val_x = pd.concat([datasetf7_1_val_x,datasetf7_2_val_x,datasetf7_3_val_x,datasetf7_5_val_x,datasetf7_7_val_x], axis=1)
dataset_x_visit = pd.concat([datasetf1_x,datasetf7_x,dataset_x_xgb,datasetfadd1_x,datasetfadd2_x,dataset1_x], axis=1)
dataset_val_x_visit = pd.concat([datasetf1_val_x,datasetf7_val_x,dataset_val_x_xgb,datasetfadd1_val_x,datasetfadd2_val_x,dataset1_val_x], axis=1)
dataset_x_image = pd.concat([dataset_x_m,dataset_x_xgb_image256,dataset1_x_image256,dataset_x_ms,dataset_x_EF,dataset_x_se_ef,dataset_x_se2,dataset_x_se_ir], axis=1)
dataset_val_x_image = pd.concat([dataset_val_x_m,dataset_val_x_xgb_image256,dataset1_val_x_image256,dataset_val_x_ms,dataset_val_x_EF,dataset_val_x_se_ef,dataset_val_x_se2,dataset_val_x_se_ir], axis=1)
dataset_x = pd.concat([dataset_x_image,dataset_x_visit], axis=1)
dataset_val_x = pd.concat([dataset_val_x_image,dataset_val_x_visit], axis=1)
dataset_y = dataset_xgb.label
dataset_val_y = dataset_val_xgb.label
dataset_x = pd.concat([dataset_x,dataset_val_x])
dataset_y = pd.concat([dataset_y,dataset_val_y])

dataset = xgb.DMatrix(dataset_x,label=dataset_y)
dataset_val = xgb.DMatrix(dataset_val_x,label=dataset_val_y)
dataset_val1 = xgb.DMatrix(dataset_val_x)
params ={'learning_rate': 0.02,
          'max_depth': 7,
          'objective': 'multi:softprob',
          'tree_method': 'gpu_hist',
          'gpu_id': 0,
          'num_class':9,
          'min_child_weight':1,
          'gamma':0,
          'subsample':0.8,
	      'colsample_bytree':0.8,
	      'seed':27
        }

watchlist = [(dataset, 'train'),(dataset_val, 'eval')]
model = xgb.train(params,dataset, num_boost_round=3500,evals=watchlist)
y_pred = model.predict(dataset_val1)
y_pred =list(y_pred)
pred_max = y_pred_max(y_pred)
class_val()
#model.save_model('model/testXGboostClass_class_val1.model')
model.save_model('model/testXGboostClass_class_val1+val2.model')

