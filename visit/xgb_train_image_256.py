import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

#输出csv文件
def write_csv(name_csv,state):
    list_class = []
    if state == 'val1':
        url_val = 'dataset/dataset_val1.csv'
        with open(url_val) as f:
            reader = csv.reader(f)
            reader_val = list(reader)
    if state == 'val2':
        url_val = 'dataset/dataset_val2.csv'
        with open(url_val) as f:
            reader = csv.reader(f)
            reader_val = list(reader)
    url_test = 'dataset/dataset_test.csv'
    with open(url_test) as f:
        reader = csv.reader(f)
        reader_test = list(reader)
    if state == 'val1' or state == 'val2':
        csv_name = ['buliding_id', 'x1(v+seresnext256)', 'x2(v+seresnext256)', 'x3(v+seresnext256)', 'x4(v+seresnext256)', 'x5(v+seresnext256)', 'x6(v+seresnext256)', 'x7(v+seresnext256)', 'x8(v+seresnext256)', 'x9(v+seresnext256)', 'predicted', 'label']
        for i in range(len(y_pred)):
            list_class += [
                [reader_val[i + 1][0], float(y_pred[i][1]), float(y_pred[i][2]), float(y_pred[i][3]),
                 float(y_pred[i][4]), float(y_pred[i][5]),
                 float(y_pred[i][6]), float(y_pred[i][7]), float(y_pred[i][8]), float(y_pred[i][0]), pred_max[i],
                 reader_val[i + 1][1]]]
        test = pd.DataFrame(columns=csv_name, data=list_class)
        test.to_csv(name_csv, encoding='gbk', index=None)
    if state == 'test':
        csv_name = ['buliding_id', 'x1(v+seresnext256)', 'x2(v+seresnext256)', 'x3(v+seresnext256)', 'x4(v+seresnext256)', 'x5(v+seresnext256)', 'x6(v+seresnext256)', 'x7(v+seresnext256)', 'x8(v+seresnext256)', 'x9(v+seresnext256)', 'predicted']
        for i in range(len(y_pred)):
            list_class += [
                [reader_test[i + 1][0], float(y_pred[i][1]), float(y_pred[i][2]), float(y_pred[i][3]),
                 float(y_pred[i][4]), float(y_pred[i][5]),
                 float(y_pred[i][6]), float(y_pred[i][7]), float(y_pred[i][8]), float(y_pred[i][0]), pred_max[i]]]
        test = pd.DataFrame(columns=csv_name, data=list_class)
        test.to_csv(name_csv, encoding='gbk', index=None)

#drop_out = ['buliding_id','label','channel102','channel219','channel138','channel70','channel211','channel254','channel107','channel180','channel181','channel100']
drop_out = ['buliding_id','label']
dataset1 = pd.read_csv('dataset/seresnext_256_train.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('dataset/seresnext_256_val2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset_val = pd.read_csv('dataset/seresnext_256_val1.csv')
dataset_val.label.replace(-1,0,inplace=True)
dataset_test = pd.read_csv('dataset/seresnext_256_test.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset_val.drop_duplicates(inplace=True)
dataset_test.drop_duplicates(inplace=True)

dataset1_y = dataset1.label
dataset1_x = dataset1.drop(drop_out,axis=1)
dataset2_y = dataset2.label
dataset2_x = dataset2.drop(drop_out,axis=1)
dataset_val_y = dataset_val.label
dataset_val_x = dataset_val.drop(drop_out,axis=1)
dataset_test_x = dataset_test.drop(['buliding_id'],axis=1)

dataset1_v = pd.read_csv('dataset/dataset_train_9to0.csv')
dataset1_v.label.replace(-1, 0, inplace=True)
dataset2_v = pd.read_csv('dataset/dataset_val2.csv')
dataset2_v.label.replace(-1, 0, inplace=True)
dataset_val_v = pd.read_csv('dataset/dataset_val1.csv')
dataset_val_v.label.replace(-1, 0, inplace=True)
dataset_test_v = pd.read_csv('dataset/dataset_test.csv')
dataset1_v.drop_duplicates(inplace=True)
dataset2_v.drop_duplicates(inplace=True)
dataset_val_v.drop_duplicates(inplace=True)
dataset_test_v.drop_duplicates(inplace=True)
dataset1_v_x = dataset1_v.drop(drop_out, axis=1)
dataset2_v_x = dataset2_v.drop(drop_out, axis=1)
dataset_val_v_x = dataset_val_v.drop(drop_out, axis=1)
dataset_test_v_x = dataset_test_v.drop(['buliding_id'], axis=1)

dataset1_x = pd.concat([dataset1_v_x,dataset1_x], axis=1)
dataset2_x = pd.concat([dataset2_v_x,dataset2_x], axis=1)
dataset_val_x = pd.concat([dataset_val_v_x,dataset_val_x], axis=1)
dataset_test_x = pd.concat([dataset_test_v_x,dataset_test_x], axis=1)

dataset_x = pd.concat([dataset1_x,dataset2_x])
dataset_y = pd.concat([dataset1_y,dataset2_y])
dataset_x_all = pd.concat([dataset1_x,dataset2_x,dataset_val_x])
dataset_y_all = pd.concat([dataset1_y,dataset2_y,dataset_val_y])
#dataset = xgb.DMatrix(dataset1_x,label=dataset1_y)
#dataset = xgb.DMatrix(dataset_x,label=dataset_y)
dataset = xgb.DMatrix(dataset_x_all,label=dataset_y_all)
dataset_val = xgb.DMatrix(dataset_val_x,label=dataset_val_y)
dataset_val1 = xgb.DMatrix(dataset_val_x)
dataset_val2 = xgb.DMatrix(dataset2_x)
dataset_test = xgb.DMatrix(dataset_test_x)

params ={'learning_rate': 0.05,
          'max_depth': 10,
          'objective': 'multi:softprob',
          'tree_method': 'gpu_hist',
          'gpu_id':0,
          'num_class':9,
          'min_child_weight':1,
          'gamma':0.05,
          'subsample':0.8,
	  'colsample_bytree':0.8,
	  'seed':27
        }

watchlist = [(dataset, 'train'),(dataset_val, 'eval')]
model = xgb.train(params,dataset, num_boost_round=2000,evals=watchlist)
#feature_score(model)
#model.save_model('model/testXGboostClass_320000(image_256).model')
#model.save_model('model/testXGboostClass_400000(image_256).model')
'''
y_pred = model.predict(dataset_val1)
y_pred =list(y_pred)
pred_max = y_pred_max(y_pred)
write_csv('dataset/class_dataset/x_val1(v+seresnext256).csv','val1')
y_pred = model.predict(dataset_val2)
y_pred =list(y_pred)
pred_max = y_pred_max(y_pred)
write_csv('dataset/class_dataset/x_val2(v+seresnext256).csv','val2')
'''
y_pred = model.predict(dataset_test)
y_pred =list(y_pred)
pred_max = y_pred_max(y_pred)
write_csv('dataset/class_dataset/x_test(v+seresnext256).csv','test')
