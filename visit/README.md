Environmental Requirement:
xgboost
numpy
csv

使用说明：
一、访问数据处理
(1)提取训练集和测试集特征
python extract_feature_train.py
python extract_feature_test.py
由于复赛的数据集有40w所以特征提取的时间很长（我们用了差不多1天），且序号不连续可以同时运行多个程序进行特征提取，最后放置在同一个csv文件下，并保存在文件夹dataset下。
（2）将训练集中的9类别都转换成0类别
python label_making(9to0).py
（3）用excel工具对训练集按building_id进行排序，取前0~319999为训练集，320000~359999为验证集1，360000~399999为验证集2，均保存在文件夹dataset下。
（4）模型训练,保存模型概率
python xgb_train.py

二、图像处理（256）
（1）需要先运行Seresnet164中的代码完成256维输出
（2）训练获得256维图像的分类概率
 python xgb_train_image_256.py
 
三、融合
（1）将要进行融合的概率和原特征的路径修改，运行训练融合xgboost
python ensamble_train.py
（2）用测试集输出最终结果
python submit_result.py