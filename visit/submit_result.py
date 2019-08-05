import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import csv

#特征贡献评分
def feature_score(model):
    # save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    fs = []
    for (key, value) in feature_score:
        fs.append("{0},{1}\n".format(key, value))
    with open('dataset/feature_score_mul.csv', 'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

#各类别预测的数量
def class_sum():
    global class_num
    num = len(pred_max)
    num_id = 0
    while (num):
        class_num[int(pred_max[num_id])] += 1
        num -= 1
        num_id += 1
    class_num[9] = class_num[0]
    num = len(class_num) - 1
    num_id = 1
    with open(userFile_class, "a", encoding="utf-8")as f:
        temp = "\n"
        f.write(temp)
        while (num):
            # print(class_num[num_id])
            temp = max_id_list[num_id] + "\t" + str(class_num[num_id]) + "\n"
            f.write(temp)
            num_id += 1
            num -= 1

#线下检测类别及精确度
def class_val():
    global class_num
    global class_num_sum
    num = len(pred_max)
    num_id = 0
    sum = 0
    while (num):
        if int(pred_max[num_id]) == int(reader_val[num_id+1][1]):
            class_num[int(pred_max[num_id])] += 1
        class_num_sum[int(reader_val[num_id+1][1])] += 1
        num -= 1
        num_id += 1
    class_num[9] = class_num[0]
    class_num_sum[9] = class_num_sum[0]
    num = len(class_num) - 1
    num_id = 1
    with open(userFile_class, "a", encoding="utf-8")as f:
        temp = "\n"
        f.write(temp)
        while (num):
            # print(class_num[num_id])
            temp = max_id_list[num_id] + "\t" + str(class_num[num_id]) + "\t" + str(round(class_num[num_id]/class_num_sum[num_id],3)) + "\n"
            sum += class_num[num_id]
            f.write(temp)
            num_id += 1
            num -= 1
        temp = 'score' + "\t" + str(round(sum / len(pred_max), 3)) + "\n"
        f.write(temp)

#生成结果文件
def result_data():
    userFile = "result_data.txt"
    Num_1 = 0
    Num_2 = 0
    Num_3 = 0
    Num_4 = 0
    Num_5 = 0
    Num_6 = 0
    num = len(pred_max)
    num_id = 0
    with open(userFile, "a", encoding="utf-8")as f:
        while (num):
            AreaID = str(Num_1) + str(Num_2) + str(Num_3) + str(Num_4) + str(Num_5) + str(Num_6)
            temp = AreaID + "\t" + max_id_list[int(pred_max[num_id])] + "\n"
            f.write(temp)
            Num_6 += 1
            if Num_6 >= 10:
                Num_5 += 1
                Num_6 = 0
                if Num_5 >= 10:
                    Num_4 += 1
                    Num_5 = 0
                    if Num_4 >= 10:
                        Num_3 += 1
                        Num_4 = 0
                        if Num_3 >= 10:
                            Num_2 += 1
                            Num_3 = 0
            num -= 1
            num_id += 1

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

list_class = []
max_id_list = ["009","001","002","003","004","005","006","007","008","009"]
class_num =[0,0,0,0,0,0,0,0,0,0]
class_num_sum =[0,0,0,0,0,0,0,0,0,0]
userFile_class = "class_num.txt"

######导入val1、val2######
##Visit--test
dataset1 = pd.read_csv('test/dataset_test.csv')
dataset1.drop_duplicates(inplace=True)
dataset1_x = dataset1.drop(['buliding_id'],axis=1)

##Visit--test(image256)
dataset1_image256 = pd.read_csv('dataset/dataset_test(image_256_32wmodel).csv')
dataset1_image256.drop_duplicates(inplace=True)
dataset1_x_image256 = dataset1_image256.drop(['buliding_id'],axis=1)

##Visit--xgbooost
dataset_xgb = pd.read_csv('dataset/visit_xgb/test_8.csv')
dataset_xgb.drop_duplicates(inplace=True)
dataset_x_xgb = dataset_xgb.drop(['buliding_id','predicted'],axis=1)

##F1_1
datasetf1_1 = pd.read_csv('dataset/visit_xgb/test_1_1.csv')
datasetf1_1.drop_duplicates(inplace=True)
datasetf1_1_x = datasetf1_1.drop(['buliding_id','predicted'],axis=1)

##F1_2
datasetf1_2 = pd.read_csv('dataset/visit_xgb/test_1_2.csv')
datasetf1_2.drop_duplicates(inplace=True)
datasetf1_2_x = datasetf1_2.drop(['buliding_id','predicted'],axis=1)

##F1_3
datasetf1_3 = pd.read_csv('dataset/visit_xgb/test_1_3.csv')
datasetf1_3.drop_duplicates(inplace=True)
datasetf1_3_x = datasetf1_3.drop(['buliding_id','predicted'],axis=1)

##F1_8
datasetf1_8 = pd.read_csv('dataset/visit_xgb/test_1_8.csv')
datasetf1_8.drop_duplicates(inplace=True)
datasetf1_8_x = datasetf1_8.drop(['buliding_id','predicted'],axis=1)

##F7_1
datasetf7_1 = pd.read_csv('dataset/visit_xgb/test_7_1.csv')
datasetf7_1.drop_duplicates(inplace=True)
datasetf7_1_x = datasetf7_1.drop(['buliding_id','predicted'],axis=1)

##F7_2
datasetf7_2 = pd.read_csv('dataset/visit_xgb/test_7_2.csv')
datasetf7_2.drop_duplicates(inplace=True)
datasetf7_2_x = datasetf7_2.drop(['buliding_id','predicted'],axis=1)

##F7_3
datasetf7_3 = pd.read_csv('dataset/visit_xgb/test_7_3.csv')
datasetf7_3.drop_duplicates(inplace=True)
datasetf7_3_x = datasetf7_3.drop(['buliding_id','predicted'],axis=1)

##F7_5
datasetf7_5 = pd.read_csv('dataset/visit_xgb/test_7_5.csv')
datasetf7_5.drop_duplicates(inplace=True)
datasetf7_5_x = datasetf7_5.drop(['buliding_id','predicted'],axis=1)

##F7_7
datasetf7_7 = pd.read_csv('dataset/visit_xgb/test_7_7.csv')
datasetf7_7.drop_duplicates(inplace=True)
datasetf7_7_x = datasetf7_7.drop(['buliding_id','predicted'],axis=1)

##Fadd1
datasetfadd1 = pd.read_csv('dataset/visit_xgb/test_add1.csv')
datasetfadd1.drop_duplicates(inplace=True)
datasetfadd1_x = datasetfadd1.drop(['buliding_id','predicted'],axis=1)

##Fadd2
datasetfadd2 = pd.read_csv('dataset/visit_xgb/test_add2.csv')
datasetfadd2.drop_duplicates(inplace=True)
datasetfadd2_x = datasetfadd2.drop(['buliding_id','predicted'],axis=1)

##图像
dataset_m = pd.read_csv('dataset/class_dataset/m_test.csv')
dataset_m.drop_duplicates(inplace=True)
dataset_x_m = dataset_m.drop(['buliding_id','predicted'],axis=1)

##图像(数据增强版)
dataset_ms = pd.read_csv('dataset/class_dataset/ms_test.csv')
dataset_ms.drop_duplicates(inplace=True)
dataset_x_ms = dataset_ms.drop(['buliding_id','predicted'],axis=1)

##image256--xgboost
dataset_xgb_image256 = pd.read_csv('dataset/class_dataset/x_test(image_256).csv')
dataset_xgb_image256.drop_duplicates(inplace=True)
dataset_x_xgb_image256 = dataset_xgb_image256.drop(['buliding_id','predicted'],axis=1)

##图像(efficientnet_fc)
dataset_EF = pd.read_csv('dataset/class_dataset/EF_test.csv')
dataset_EF.drop_duplicates(inplace=True)
dataset_x_EF = dataset_EF.drop(['buliding_id','predicted'],axis=1)

##图像(se_ef)
dataset_se_ef = pd.read_csv('dataset/class_dataset/se_ef_test.csv')
dataset_se_ef.drop_duplicates(inplace=True)
dataset_x_se_ef = dataset_se_ef.drop(['buliding_id','predicted'],axis=1)

##图像(se2)
dataset_se2 = pd.read_csv('dataset/class_dataset/se2_test.csv')
dataset_se2.drop_duplicates(inplace=True)
dataset_x_se2 = dataset_se2.drop(['buliding_id','predicted'],axis=1)

##图像(se_ir)
dataset_se_ir = pd.read_csv('dataset/class_dataset/se_ir_test.csv')
dataset_se_ir.drop_duplicates(inplace=True)
dataset_x_se_ir = dataset_se_ir.drop(['buliding_id','predicted'],axis=1)

##访问+256(seresnext)
dataset_v256x = pd.read_csv('dataset/class_dataset/x_test(v+seresnext256).csv')
dataset_v256x.drop_duplicates(inplace=True)
dataset_x_v256x = dataset_v256x.drop(['buliding_id','predicted'],axis=1)

datasetf1_x = pd.concat([datasetf1_1_x,datasetf1_2_x,datasetf1_3_x,datasetf1_8_x], axis=1)
datasetf7_x = pd.concat([datasetf7_1_x,datasetf7_2_x,datasetf7_3_x,datasetf7_5_x,datasetf7_7_x], axis=1)
dataset_x_visit = pd.concat([datasetf1_x,datasetf7_x,dataset_x_xgb,datasetfadd1_x,datasetfadd2_x,dataset1_x], axis=1)
dataset_x_image = pd.concat([dataset_x_m,dataset_x_xgb_image256,dataset1_x_image256,dataset_x_ms,dataset_x_EF,dataset_x_se_ef,dataset_x_se2,dataset_x_se_ir], axis=1)
dataset_x = pd.concat([dataset_x_image,dataset_x_visit], axis=1)
dataset_test = xgb.DMatrix(dataset_x)
url_val = 'dataset/dataset_val1.csv'
with open(url_val) as f:
    reader = csv.reader(f)
    reader_val = list(reader)

model = xgb.Booster(model_file='model/testXGboostClass_class_val1+val2.model')
y_pred = model.predict(dataset_test)
y_pred =list(y_pred)
pred_max = y_pred_max(y_pred)

#online
class_sum()
result_data()
#feature_score(model)