import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv

##feature group
buliding = ['buliding_id']
week_F = ["weekday1", "weekday2", "weekday3", "weekday4", "weekday5", "weekday6", "weekday7", "holiday_num_ratio",
          'weekend_to_weekday', "week_start_to_finish_ratio"]
mouth_F = ['March_to_December', "morethan7_P_March_ratio", "morethan3_P_January_ratio", "spring_trip_P_ratio",
           "spring_ratio", "Nationalday_weekday_ratio",
           "Nationalday_weekend_ratio", "Nationalday_ratio", "NewYearday_ratio", "most_multiple_February",
           "less_multiple_February", "work_to_relax",
           'day30_ratio', 'day25to30']
mouth_ratio_F = ['mouth10_ratio', 'mouth11_ratio', 'mouth12_ratio', 'mouth1_ratio', 'mouth2_ratio', 'mouth3_ratio']
hourlong_F = ["time_lessthan2_ratio", "time_morethan7_ratio", "time_after_18_ratio", "ave_time_daily",
              "P_6to8_ratio", "P_8to18_ratio", "P_visithour14_ratio"]
hour_F = ["most_P_hour", "ave_early_hour", "ave_leave_hour", "most_multiple_7to10", "most_multiple_11to13",
          "most_multiple_13to15", "most_multiple_17to20",
          "less_multiple_7to10", "less_multiple_11to13", "less_multiple_13to15", "less_multiple_17to20",
          'high_hour_hosbital', 'high_hour_eating', 'high_hour_num',
          'mutation_hour', 'high_hour_night']
hour_ratio_F = ["hour_ratio0", "hour_ratio1", "hour_ratio2", "hour_ratio3", "hour_ratio4", "hour_ratio5",
                "hour_ratio6",
                "hour_ratio7", "hour_ratio8", "hour_ratio9", "hour_ratio10", "hour_ratio11", "hour_ratio12",
                "hour_ratio13", "hour_ratio14", "hour_ratio15",
                "hour_ratio16", "hour_ratio17", "hour_ratio18", "hour_ratio19", "hour_ratio20", "hour_ratio21",
                "hour_ratio22", "hour_ratio23"]
sum_F = ['peron_sum', 'visit_sum', 'ave_visit_P', "P_visit_more_num", "P_visit_more_ratio"]
user_F = ['visit_morethan2_ratio', 'visit_morethan10_ratio', 'visit_morethan30_ratio', "same_P_mostday",
          "ave_timegap", "morethan3day_P_num", "morethan3day_P_ratio",
          "morethan48hour_P_num", "morethan72hour_P_num"]

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
def write_csv_val(test_num,y_pred1,pred_max1,y_pred2,pred_max2):
    name_csv1 = 'dataset/visit_xgb/val1'
    name_csv2 = 'dataset/visit_xgb/val2'
    csv_name_F = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    name_csv1 = name_csv1 + test_num + '.csv'
    name_csv2 = name_csv2 + test_num + '.csv'
    for i in range(len(csv_name_F)):
        csv_name_F[i] = csv_name_F[i] + test_num
    list_class1 =[]
    list_class2 = []
    csv_name = ['buliding_id',csv_name_F[0], csv_name_F[1], csv_name_F[2], csv_name_F[3], csv_name_F[4], csv_name_F[5], csv_name_F[6], csv_name_F[7], csv_name_F[8], 'predicted', 'label']
    url_val = 'dataset/dataset_val1.csv'
    with open(url_val) as f:
        reader = csv.reader(f)
        reader_val1 = list(reader)
    url_val = 'dataset/dataset_val2.csv'
    with open(url_val) as f:
        reader = csv.reader(f)
        reader_val2 = list(reader)
    for i in range(len(y_pred1)):
        list_class1 += [
                [reader_val1[i + 1][0], float(y_pred1[i][1]), float(y_pred1[i][2]), float(y_pred1[i][3]),
                 float(y_pred1[i][4]), float(y_pred1[i][5]),
                 float(y_pred1[i][6]), float(y_pred1[i][7]), float(y_pred1[i][8]), float(y_pred1[i][0]), pred_max1[i],
                 reader_val1[i + 1][1]]]
    test = pd.DataFrame(columns=csv_name, data=list_class1)
    test.to_csv(name_csv1, encoding='gbk', index=None)
    for i in range(len(y_pred2)):
        list_class2 += [
                [reader_val2[i + 1][0], float(y_pred2[i][1]), float(y_pred2[i][2]), float(y_pred2[i][3]),
                 float(y_pred2[i][4]), float(y_pred2[i][5]),
                 float(y_pred2[i][6]), float(y_pred2[i][7]), float(y_pred2[i][8]), float(y_pred2[i][0]), pred_max2[i],
                 reader_val2[i + 1][1]]]
    test = pd.DataFrame(columns=csv_name, data=list_class2)
    test.to_csv(name_csv2, encoding='gbk', index=None)

#输出csv文件
def write_csv_test(test_num, y_pred1, pred_max1):
    name_csv1 = 'dataset/visit_xgb/test'
    csv_name_F = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
    name_csv1 = name_csv1 + test_num + '.csv'
    for i in range(len(csv_name_F)):
        csv_name_F[i] = csv_name_F[i] + test_num
    list_class1 =[]
    csv_name = ['buliding_id',csv_name_F[0], csv_name_F[1], csv_name_F[2], csv_name_F[3], csv_name_F[4], csv_name_F[5], csv_name_F[6], csv_name_F[7], csv_name_F[8], 'predicted']
    url_val = 'dataset/dataset_test.csv'
    with open(url_val) as f:
        reader = csv.reader(f)
        reader_val1 = list(reader)
    for i in range(len(y_pred1)):
        list_class1 += [
                [reader_val1[i + 1][0], float(y_pred1[i][1]), float(y_pred1[i][2]), float(y_pred1[i][3]),
                 float(y_pred1[i][4]), float(y_pred1[i][5]),
                 float(y_pred1[i][6]), float(y_pred1[i][7]), float(y_pred1[i][8]), float(y_pred1[i][0]), pred_max1[i]]]
    test = pd.DataFrame(columns=csv_name, data=list_class1)
    test.to_csv(name_csv1, encoding='gbk', index=None)


def model_set(drop_out,test_num,state):
    label_dropout = ['label']
    #drop_out = drop_out + ['hour22wk','hour18wk']
    drop_out_Alabel = drop_out + label_dropout
    dataset1 = pd.read_csv('dataset/dataset_train_9to0.csv')
    dataset1.label.replace(-1, 0, inplace=True)
    dataset2 = pd.read_csv('dataset/dataset_val2.csv')
    dataset2.label.replace(-1, 0, inplace=True)
    dataset_val = pd.read_csv('dataset/dataset_val1.csv')
    dataset_val.label.replace(-1, 0, inplace=True)
    dataset_test = pd.read_csv('dataset/dataset_test.csv')
    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset_val.drop_duplicates(inplace=True)
    dataset_test.drop_duplicates(inplace=True)
    dataset1_y = dataset1.label
    dataset1_x = dataset1.drop(drop_out_Alabel, axis=1)
    dataset2_y = dataset2.label
    dataset2_x = dataset2.drop(drop_out_Alabel, axis=1)
    dataset_val_y = dataset_val.label
    dataset_val_x = dataset_val.drop(drop_out_Alabel, axis=1)
    dataset_test_x = dataset_test.drop(drop_out, axis=1)

    dataset_x_all = pd.concat([dataset1_x, dataset2_x, dataset_val_x])
    dataset_y_all = pd.concat([dataset1_y, dataset2_y, dataset_val_y])
    if state == 'val':
        dataset = xgb.DMatrix(dataset1_x, label=dataset1_y)
    if state == 'test':
        dataset = xgb.DMatrix(dataset_x_all,label=dataset_y_all)
    dataset_val = xgb.DMatrix(dataset_val_x, label=dataset_val_y)
    dataset_val1 = xgb.DMatrix(dataset_val_x)
    dataset_val2 = xgb.DMatrix(dataset2_x)
    dataset_test = xgb.DMatrix(dataset_test_x)

    params = {'learning_rate': 0.05,
              'max_depth': 10,
              'objective': 'multi:softprob',
              'tree_method': 'gpu_hist',
              'gpu_id': 2,
              'num_class': 9,
              'min_child_weight': 1,
              'gamma': 0.05,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'seed': 27
              }
    watchlist = [(dataset, 'train'), (dataset_val, 'eval')]
    model = xgb.train(params, dataset, num_boost_round=3500, evals=watchlist)
    if state == 'val':
        y_pred1 = model.predict(dataset_val1)
        y_pred1 = list(y_pred1)
        pred_max1 = y_pred_max(y_pred1)
        y_pred2 = model.predict(dataset_val2)
        y_pred2 = list(y_pred2)
        pred_max2 = y_pred_max(y_pred2)
        model_name = 'model/testXGboostClass_320000' + test_num + '.model'
        model.save_model(model_name)
        write_csv_val(test_num, y_pred1, pred_max1, y_pred2, pred_max2)
        print('finish val'+ test_num)
    if state == 'test':
        y_pred1 = model.predict(dataset_test)
        y_pred1 = list(y_pred1)
        pred_max1 = y_pred_max(y_pred1)
        model_name = 'model/testXGboostClass_400000' + test_num + '.model'
        model.save_model(model_name)
        write_csv_test(test_num, y_pred1, pred_max1)
        print('finish test' + test_num)

if __name__ == '__main__':
    '''
    #drop_out = buliding + user_F + sum_F + hour_ratio_F + hour_F + hourlong_F + mouth_ratio_F + mouth_F + week_F
    #单特征组模型1
    drop_out = buliding + sum_F + hour_ratio_F + hour_F + hourlong_F + mouth_ratio_F + mouth_F + week_F
    model_set(drop_out, '_1_1', 'val')
    model_set(drop_out, '_1_1', 'test')
    #单特征组模型2
    drop_out = buliding + user_F + hour_ratio_F + hour_F + hourlong_F + mouth_ratio_F + mouth_F + week_F
    model_set(drop_out, '_1_2', 'val')
    model_set(drop_out, '_1_2', 'test')
    #单特征组模型3
    drop_out = buliding + user_F + sum_F + hour_F + hourlong_F + mouth_ratio_F + mouth_F + week_F
    model_set(drop_out, '_1_3', 'val')
    model_set(drop_out, '_1_3', 'test')
    #单特征组模型8
    drop_out = buliding + user_F + sum_F + hour_ratio_F + hour_F + hourlong_F + mouth_ratio_F + mouth_F
    model_set(drop_out, '_1_8', 'val')
    model_set(drop_out, '_1_8', 'test')
    #七特征组模型1
    drop_out = buliding + user_F
    model_set(drop_out, '_7_1', 'val')
    model_set(drop_out, '_7_1', 'test')
    '''
    #七特征组模型2
    drop_out = buliding + sum_F
    #model_set(drop_out, '_7_2', 'val')
    model_set(drop_out, '_7_2', 'test')
    #七特征组模型3
    drop_out = buliding + hour_ratio_F
    model_set(drop_out, '_7_3', 'val')
    model_set(drop_out, '_7_3', 'test')
    #七特征组模型5
    drop_out = buliding + hourlong_F
    model_set(drop_out, '_7_5', 'val')
    model_set(drop_out, '_7_5', 'test')
    #七特征组模型7
    drop_out = buliding + mouth_F
    model_set(drop_out, '_7_7', 'val')
    model_set(drop_out, '_7_7', 'test')
    #全特征组模型
    drop_out = buliding
    model_set(drop_out, '_8', 'val')
    model_set(drop_out, '_8', 'test')