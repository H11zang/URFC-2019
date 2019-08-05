# -- coding: utf-8 --
#####特征说明#####
#buliding_id   建筑物标号/图片标号
#label 建筑物所属类别
#person_sum 总访问人数
#visit_sum 总访问次数
#ave_visit_P 人均访问次数
#visit_morethan2_ratio 访问次数大于等于2次的人数占总人数的比例
#visit_morethan10_ratio 访问次数大于等于10次的人数占总人数的比例
#visit_morethan30_ratio 访问次数大于等于30次的人数占总人数的比例
#same_P_mostday 同一个访问者访问的最多天数
#ave_time_daily 单天平均访问时长
#time_after_18_ratio 18点后来访的人数占比
#time_lessthan2_ratio 所待时间低于3小时的人数占比
#time_morethan7_ratio 所待时间长于7小时的人数占比
#weekday1 周一的人数占本周访问的比例
#weekday2 周二的人数占本周访问的比例
#weekday3 周三的人数占本周访问的比例
#weekday4 周四的人数占本周访问的比例
#weekday5 周五的人数占本周访问的比例
#weekday6 周六的人数占本周访问的比例
#weekday7 周天的人数占本周访问的比例
#holiday_num_ratio 节假日的访问人数所占比例
#most_P_hour 人数最多的时刻
#ave_early_hour 最早访问时刻平均值
#ave_leave_hour 离开时刻平均值
#morethan7_P_March_ratio 同一个访问者在三月访问次数超过7次的占三月访问总人数比例
#ave_timegap 同一个访问者访问的平均间隔（访次数低于30次）
#morethan3_P_January_ratio 同一个访问者在一月访问次数超过3次的占一月访问总人数比例
#morethan3day_P_num 同一访问者连续访问达24小时时长超过3天的人数
#morethan3day_P_ratio 同一访问者连续访问达24小时时长超过3天的人数所占总人数的比例
#morethan48hour_P_num 访问时长超过48小时的人数
#morethan72hour_P_num 访问时长超过72小时的人数
#spring_trip_P_ratio 春运期间（1月29日~2月12日）的访问人数占2019年访问总人数的比例
#P_6to8_ratio 6点到10点的访问量占总人数的比例
#P_8to18_ratio 8点到18点的访问量占总人数的比例
#P_visithour14_ratio 14点进行访问的人数占总人数的比例
#P_visit_more_num 一天内多次访问的人数
#P_visit_more_ratio 一天内多次访问的人数占总人数的比例
#most_multiple_February 2月15日~2月28日某一天的人数是前一天的倍数（开学报到）
#less_multiple_February 2月15日~2月28日因为开学人数骤减
#hour[] 24小时中各自的访问人数
#hour_ratio[] 24小时中各自的访问人数平均访问占比
#work_to_relax 上班期间（8点到17点）与下班期间（18点到23点）人数比值
#most_multiple_7to10&less_multiple_7to10 7点到10点之间人员骤增与骤减（倍数）
#most_multiple_11to13&less_multiple_11to13 11点到13点之间人员骤增与骤减（倍数）
#most_multiple_13to15&less_multiple_13to15 13点到15点之间人员骤增与骤减（倍数）
#most_multiple_17to20&less_multiple_17to20 17点到20点之间人员骤增与骤减（倍数）
#spring_ratio 2月4到6日与2月1到3日的访问量比值（春节前后比）
#Nationalday_weekday_ratio 10月1日与10月8到12日的访问量的均值的比值（国庆与工作日对比）
#Nationalday_weekend_ratio 10月1日与10月13到14日的访问量的均值的比值（国庆与周末对比）
#Nationalday_ratio 10月1日到3日的人流量均值和10月4日到7日的均值的比值（黄金周的前几天比后几天的人多）
#week_start_to_finish_ratio 十一月周一周二和周三周四的人数比值（医院星期前的人数比之后的多）
#NewYearday_ratio 12月27日到29日与12月30日到1月1日的访问量比值（元旦前后比）
#mouth_ratio 月度访问信息比较（10月到4月）
#March_to_December 3月的人流量均值与12月人流量均值比较（判断人流量的季节性变化）
#high_hour_hosbital 早上10点的人数和下午16点的人是否为时间峰值（0和1）
#high_hour_eating 早上11~13点的人数和下午18~20点的人是否为时间峰值（0和1）
#high_hour_num 峰值数量
#mutation_hour 是否有骤增骤减
#high_hour_night 20点后是否有峰值出现（0和1）
#weekend_to_weekday 周末人均和工作日人均比值（取十一月）
#####模块引用#####
import pandas as pd
import os
import datetime
root = "0" #设置文件夹
txt_start = 0 #设置从第几个文件开始
txt_stop = 24999#设置从第几个文件结束
csv_N = 'dataset_train.csv'#保持csv文件名称
peron_sum = 0
visit_sum = 0
hour_sum = 0
visit_sum_2019 = 0
time_sum = 0
visit_morethan2 = 0
visit_morethan10 = 0
visit_morethan30 = 0
same_P_mostday = 0
ave_time_daily = 0
time_lessthan2 = 0
time_morethan7 = 0
time_after_18 = 0
weekday1 = 0
weekday2 = 0
weekday3 = 0
weekday4 = 0
weekday5 = 0
weekday6 = 0
weekday7 = 0
weekday1_num = 0
weekday2_num = 0
weekday3_num = 0
weekday4_num = 0
weekday5_num = 0
weekday6_num = 0
weekday7_num = 0
holiday_num = 0
hour = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
multiple_hour = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
hour_ratio = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
hour_mul = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
most_multiple_7to10 = 0
less_multiple_7to10 = 1
most_multiple_11to13 = 0
less_multiple_11to13 = 1
most_multiple_13to15 = 0
less_multiple_13to15 = 1
most_multiple_17to20 = 0
less_multiple_17to20 = 1
mouth = [31,31,28,31,30,31,30,31,31,30,31,30,31]
most_P_hour = 0
most_P_hour_2 = 0
most_P_hour_num = 0
ave_early_hour = 0
early_hour_sum = 0
ave_leave_hour = 0
leave_hour_sum = 0
morethan7_P_March_num = 0
P_March_num = 0
ave_timegap = 0
ave_timegap_num = 0
train_timenum = 0
morethan3_P_January_num = 0
P_January_num = 0
morethan3day_P_num = 0
morethan3day_P_ratio = 0
morethan48hour_P_num = 0
morethan72hour_P_num = 0
spring_trip_P_num = 0
spring_trip_P_ratio = 0
P_6to8_num = 0
P_6to8_ratio = 0
P_8to18_num = 0
P_8to18_ratio = 0
P_visithour14_num = 0
P_visithour14_ratio = 0
P_visit_more_num = 0
morethan7_P_March_ratio = 0
P_visit_more_ratio = 0
multiple_February = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
February_people =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
most_multiple_February = 0
less_multiple_February = 1
work_to_relax = 0
work_8to17 = 0
relax_18to23 = 0
spring_num = [0,0,0,0,0,0]
spring_ratio = 0
Nationalday_weekday_ratio = 0
Nationalday_weekend_ratio = 0
Nationalday_ratio = 0
October_num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
week_start_to_finish_ratio = 0
November_week_num =[0,0]
NewYearday_ratio = 0
NewYearday_num =[0,0]
mouth_num = [0,0,0,0,0,0]
mouth_ratio = [0,0,0,0,0,0]
March_to_December = 0
high_hour_hosbital = 0
high_hour_eating = 0
high_hour_P = [0,0,0,0,0,0,0,0,0,0]
high_hour_num = 0
mutation_hour = 0
high_hour_night = 0
weekend_to_weekday = 1
weekday_N_num = 0
weekend_N_num = 0
day = [0]*100
day30_ratio = 0
day25to30 = 0
morethan3_P_January_ratio = 0
csv_name = ['buliding_id','label', 'peron_sum','visit_sum','ave_visit_P','visit_morethan2_ratio','visit_morethan10_ratio',
            'visit_morethan30_ratio',"same_P_mostday","ave_time_daily","time_lessthan2_ratio","time_morethan7_ratio","time_after_18_ratio",
            "weekday1","weekday2","weekday3","weekday4","weekday5","weekday6","weekday7","holiday_num_ratio","most_P_hour","ave_early_hour",
            "ave_leave_hour","morethan7_P_March_ratio","ave_timegap","morethan3_P_January_ratio","morethan3day_P_num","morethan3day_P_ratio","morethan48hour_P_num",
             "morethan72hour_P_num","spring_trip_P_ratio","P_6to8_ratio","P_8to18_ratio","P_visithour14_ratio","P_visit_more_num","P_visit_more_ratio",
             "most_multiple_February","less_multiple_February","hour_ratio0","hour_ratio1","hour_ratio2","hour_ratio3","hour_ratio4","hour_ratio5","hour_ratio6",
             "hour_ratio7","hour_ratio8","hour_ratio9","hour_ratio10","hour_ratio11","hour_ratio12","hour_ratio13","hour_ratio14","hour_ratio15",
             "hour_ratio16","hour_ratio17","hour_ratio18","hour_ratio19","hour_ratio20","hour_ratio21","hour_ratio22","hour_ratio23","work_to_relax",
             "most_multiple_7to10","most_multiple_11to13","most_multiple_13to15","most_multiple_17to20","less_multiple_7to10",
             "less_multiple_11to13","less_multiple_13to15","less_multiple_17to20","spring_ratio","Nationalday_weekday_ratio","Nationalday_weekend_ratio","Nationalday_ratio",
             "week_start_to_finish_ratio","NewYearday_ratio",'mouth10_ratio','mouth11_ratio','mouth12_ratio','mouth1_ratio','mouth2_ratio','mouth3_ratio','March_to_December',
            'high_hour_hosbital','high_hour_eating','high_hour_num','mutation_hour','high_hour_night','weekend_to_weekday','day30_ratio','day25to30']
ret = []
list = []

#历遍文件夹
def findtxt(path, ret):
    print('开始遍历文件夹')
    n = 0
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        n = n+1
        print(n)
        if os.path.isfile(de_path):
            if de_path.endswith(".txt"):
                ret.append(de_path)
        else:
            findtxt(de_path, ret)

#输出csv文件
def write_csv():
    test = pd.DataFrame(columns=csv_name, data=list)
    test.to_csv(csv_N, encoding='gbk', index=None)

# 不同访问次数的人数计算
def visit_num(line_date_list):
    global visit_sum
    global visit_morethan2
    global visit_morethan10
    global visit_morethan30
    global same_P_mostday
    visit_sum += len(line_date_list)
    if len(line_date_list) > 2:
        visit_morethan2 += 1
    if len(line_date_list) > 10:
        visit_morethan10 += 1
    if len(line_date_list) > 30:
        visit_morethan30 += 1
    if len(line_date_list) - 1 > same_P_mostday:
        same_P_mostday = len(line_date_list) - 1

# 不同时刻访问时长计算
def time_long(line,line_date_list):
    global time_sum
    global time_lessthan2
    global time_morethan7
    global time_after_18
    time_sum += line.count('&') + line.count('|')
    for time_find in line_date_list:
        if time_find.count('|18')+time_find.count('|19')+time_find.count('|20')+time_find.count('|21')+time_find.count('|22')+time_find.count('|23') >= 1:
            time_after_18 += 1
        if time_find.count('|') < 2:
            time_lessthan2 += 1
        if time_find.count('|') >= 7:
            time_morethan7 += 1

#访问人数星期分布及十一假期人数
def day_num(line_date_list):
    global test_num
    global weekday1_num
    global weekday2_num
    global weekday3_num
    global weekday4_num
    global weekday5_num
    global weekday6_num
    global weekday7_num
    global holiday_num
    global weekday_N_num
    global weekend_N_num
    for time_find in line_date_list:
        all_data = time_find.split("&")
        d1 = datetime.date(int(all_data[0][:4]), int(all_data[0][4:6]), int(all_data[0][6:]))
        if int(all_data[0][4:6]) == 11:
            if d1.weekday() == 0 or d1.weekday() == 6:
                weekend_N_num += 1
            if d1.weekday() == 1 or d1.weekday() == 2 or d1.weekday() == 3 or d1.weekday() == 4 or d1.weekday() == 5:
                weekday_N_num += 1
        if d1.weekday() == 0:
            weekday7_num += 1
        if d1.weekday() == 1:
            weekday1_num += 1
        if d1.weekday() == 2:
            weekday2_num += 1
        if d1.weekday() == 3:
            weekday3_num += 1
        if d1.weekday() == 4:
            weekday4_num += 1
        if d1.weekday() == 5:
            weekday5_num += 1
        if d1.weekday() == 6:
            weekday6_num += 1
        if int(all_data[0][4:6]) == 10 and int(all_data[0][6:]) <= 7:
            holiday_num += 1
        if int(all_data[0][4:6]) == 12 and (int(all_data[0][6:]) == 30 or int(all_data[0][6:]) == 31):
            holiday_num += 1
        if int(all_data[0][4:6]) == 1 and int(all_data[0][6:]) == 1:
            holiday_num += 1
        if int(all_data[0][4:6]) == 2 and (int(all_data[0][6:]) >= 4 and int(all_data[0][6:]) <= 10):
            holiday_num += 1
        if int(all_data[0][4:6]) == 2 and int(all_data[0][6:]) == 19:
            holiday_num += 1
        if int(all_data[0][4:6]) == 4 and (int(all_data[0][6:]) >= 5 and int(all_data[0][6:]) <= 7):
            holiday_num += 1

#最早时刻，离开时刻均值，人数最多的时刻
def time_num(line_date_list):
    global hour
    global hour_sum
    global early_hour_sum
    global leave_hour_sum
    for time_find in line_date_list:
        all_data = time_find.split("&")
        hour_data = all_data[1].split("|")
        early_hour_sum += int(hour_data[0])
        leave_hour_sum += int(hour_data[len(hour_data)-1])
        for hour_find in hour_data:
            hour_sum += 1
            hour[int(hour_find)] += 1

#三月访问超过7次的人数比例
def march_num(line_date_list):
    global P_March_num
    global morethan7_P_March_num
    same_P_March = 0
    for time_find in line_date_list:
        if int(time_find[4:6]) == 3:
            P_March_num += 1
            same_P_March += 1
        if same_P_March == 7:
            morethan7_P_March_num += 1

#一月访问超过3次的人数比例
def January_num(line_date_list):
    global P_January_num
    global morethan3_P_January_num
    same_P_January = 0
    for time_find in line_date_list:
        if int(time_find[4:6]) == 1:
            P_January_num += 1
            same_P_January += 1
        if same_P_January == 3:
            morethan3_P_January_num += 1

#同一个访问者访问的平均间隔（访次数低于30次）
def visit_ave(line_date_list):
    global ave_timegap
    global ave_timegap_num
    global mouth
    date_sum = 0
    if int(line_date_list[len(line_date_list)-1][4:6]) < 5:
        d2 = int(line_date_list[len(line_date_list)-1][4:6]) + 12
    if int(line_date_list[len(line_date_list) - 1][4:6]) > 9:
        d2 = int(line_date_list[len(line_date_list) - 1][4:6])
    if d2 - int(line_date_list[0][4:6]) < 3:
        ave_timegap_num += 1
        if d2 - int(line_date_list[0][4:6]) == 0:
            date_sum = int(line_date_list[len(line_date_list)-1][6:8]) - int(line_date_list[0][6:8])
        if d2 - int(line_date_list[0][4:6]) == 1:
            date_sum = int(line_date_list[len(line_date_list)-1][6:8]) - int(line_date_list[0][6:8]) + mouth[int(line_date_list[0][4:6])]
        if d2 - int(line_date_list[0][4:6]) == 2:
            date_sum = int(line_date_list[len(line_date_list)-1][6:8]) - int(line_date_list[0][6:8]) + mouth[int(line_date_list[0][4:6])] + mouth[int(line_date_list[len(line_date_list)-1][4:6])-1]
        ave_timegap += date_sum / len(line_date_list)

#连续24小时访问者情况
def more_than3day(line_date_list):
    global morethan3day_P_num
    global morethan72hour_P_num
    global morethan48hour_P_num
    num = 0
    day_num = 0
    allow = 0
    day1 = 0
    day2 = 0
    day3 = 0
    for time_find in line_date_list:
        if time_find.count("|") >= 23:
            num += 1
            day_num += 1
            if day_num == 1:
                day1 = int(time_find[6:8])
            if day_num == 2:
                day2 = int(time_find[6:8])
            if day_num == 3:
                day3 = int(time_find[6:8])
            if day2 == 1:
                day2 += mouth[int(line_date_list[0][4:6])-1]
            if day3 == 2:
                day3 += mouth[int(line_date_list[0][4:6])-1]
            if day2 - day1 != 1 and day_num == 2:
                day_num = 1
                day1 = day2
                day2 = 0
            if day2 - day1 == 1 and day_num == 2:
                morethan48hour_P_num += 1
            if day3 - day2 != 1 and day_num == 3:
                day_num = 1
                day1 = day3
                day2 = 0
                day3 = 0
            if day3 - day2 == 1 and day_num == 3:
                morethan72hour_P_num += 1
        if num == 3:
            morethan3day_P_num += 1

#春运期间
def spring_trip(line_date_list):
    global visit_sum_2019
    global spring_trip_P_num
    for time_find in line_date_list:
        if int(time_find[:4]) == 2019:
            visit_sum_2019 += 1
        if int(time_find[4:6]) == 2 and int(time_find[6:8]) <= 12:
            spring_trip_P_num += 1
        if int(time_find[4:6]) == 1 and int(time_find[6:8]) >= 29:
            spring_trip_P_num += 1

#时段访问
def hour_long(line_date_list):
    global P_6to8_num
    global P_8to18_num
    global P_visithour14_num
    global P_visit_more_num
    P_6to8_allow_num = 1
    P_8to18_allow_num = 1
    visit_more_allow = 0
    hour = 0
    num = 0
    for time_find in line_date_list:
        all_data = time_find.split("|")
        num = len(all_data)
        for hour_find in all_data:
            num -= 1
            if hour_find.count("&") != 0:
                if int(hour_find[9:]) < 6 or int(hour_find[9:]) > 8:
                    P_6to8_allow_num = 0
                if int(hour_find[9:]) < 8 or int(hour_find[9:]) > 18:
                    P_8to18_allow_num = 0
                hour = int(hour_find[9:])
            if hour_find.count("&") == 0:
                if int(hour_find) < 6 or int(hour_find) > 8:
                    P_6to8_allow_num = 0
                if int(hour_find) < 8 or int(hour_find) > 18:
                    P_8to18_allow_num = 0
                if P_6to8_allow_num == 1 and num == 0:
                    P_6to8_num += 1
                if P_8to18_allow_num == 1 and num == 0:
                    P_8to18_num += 1
                if int(hour_find) - hour != 1:
                    visit_more_allow = 1
                if int(hour_find) - hour == 1:
                    hour = int(hour_find)
                if visit_more_allow == 1 and num == 0:
                    P_visit_more_num += 1
        P_6to8_allow_num = 1
        P_8to18_allow_num = 1
        visit_more_allow = 0
        hour = 0
        if int(all_data[0][9:]) == 14:
            P_visithour14_num += 1

#人数爆增检测
def February_num(line_date_list):
    global February_people
    same_P_January = 0
    for time_find in line_date_list:
        if int(time_find[4:6]) == 2 and int(time_find[6:8]) >= 15:
            February_people[int(time_find[6:8])-15] += 1

def most_multiple_day():
    global February_people
    global multiple_February
    global most_multiple_February
    global less_multiple_February
    num = len(February_people)
    num_id = 0
    while(num):
        if February_people[num_id] != 0:
            multiple_February[num_id] = February_people[num_id+1] / February_people[num_id]
            if most_multiple_February < multiple_February[num_id]:
                most_multiple_February = multiple_February[num_id]
            if less_multiple_February > multiple_February[num_id] and multiple_February[num_id] != 0:
                less_multiple_February = multiple_February[num_id]
        num -= 1
        num_id += 1

def most_multiple_hour():
    global hour
    global multiple_hour
    global most_multiple_7to10
    global less_multiple_7to10
    global most_multiple_11to13
    global less_multiple_11to13
    global most_multiple_13to15
    global less_multiple_13to15
    global most_multiple_17to20
    global less_multiple_17to20
    num = len(hour) - 1
    num_id = 0
    while(num):
        if hour[num_id] != 0:
            multiple_hour[num_id] = hour[num_id+1] / hour[num_id]
            if num_id >=7 and num_id < 10:
                if most_multiple_7to10 < multiple_hour[num_id]:
                    most_multiple_7to10 = multiple_hour[num_id]
                if less_multiple_7to10 > multiple_hour[num_id] and multiple_hour[num_id] != 0:
                    less_multiple_7to10 = multiple_hour[num_id]
            if num_id >=11 and num_id < 13:
                if most_multiple_11to13 < multiple_hour[num_id]:
                    most_multiple_11to13 = multiple_hour[num_id]
                if less_multiple_11to13 > multiple_hour[num_id] and multiple_hour[num_id] != 0:
                    less_multiple_11to13 = multiple_hour[num_id]
            if num_id >=13 and num_id < 15:
                if most_multiple_13to15 < multiple_hour[num_id]:
                    most_multiple_13to15 = multiple_hour[num_id]
                if less_multiple_13to15 > multiple_hour[num_id] and multiple_hour[num_id] != 0:
                    less_multiple_13to15 = multiple_hour[num_id]
            if num_id >=17 and num_id < 20:
                if most_multiple_17to20 < multiple_hour[num_id]:
                    most_multiple_17to20 = multiple_hour[num_id]
                if less_multiple_17to20 > multiple_hour[num_id] and multiple_hour[num_id] != 0:
                    less_multiple_17to20 = multiple_hour[num_id]
        num -= 1
        num_id += 1

#假节日比值
def day_ratio(line_date_list):
    global spring_num
    global October_num
    global November_week_num
    global NewYearday_num
    for time_find in line_date_list:
        if int(time_find[4:6]) == 2 and int(time_find[6:8]) <= 6:
            spring_num[int(time_find[6:8])-1] += 1
        if int(time_find[4:6]) == 10 and int(time_find[6:8]) <= 14:
            October_num[int(time_find[6:8])-1] += 1
        if int(time_find[4:6]) == 11:
            if int(time_find[6:8]) == 5 or int(time_find[6:8]) == 6 or int(time_find[6:8]) == 12 or int(time_find[6:8]) == 13 or int(time_find[6:8]) == 19 or int(time_find[6:8]) == 20 or int(time_find[6:8]) == 26 or int(time_find[6:8]) == 27:
                November_week_num[0] += 1
            if int(time_find[6:8]) == 7 or int(time_find[6:8]) == 8 or int(time_find[6:8]) == 14 or int(time_find[6:8]) == 15 or int(time_find[6:8]) == 21 or int(time_find[6:8]) == 22 or int(time_find[6:8]) == 28 or int(time_find[6:8]) == 29:
                November_week_num[1] += 1
        if int(time_find[4:6]) == 12:
            if int(time_find[6:8]) == 27 or int(time_find[6:8]) == 28 or int(time_find[6:8]) == 29:
                NewYearday_num[1] += 1
            if int(time_find[6:8]) == 30 or int(time_find[6:8]) == 31:
                NewYearday_num[0] += 1
        if int(time_find[4:6]) == 1 and int(time_find[6:8]) == 1:
            NewYearday_num[0] += 1

#月访问人数
def mouth_num_d(line_date_list):
    global mouth_num
    for time_find in line_date_list:
        if int(time_find[4:6]) == 10:
            mouth_num[0] += 1
        if int(time_find[4:6]) == 11:
            mouth_num[1] += 1
        if int(time_find[4:6]) == 12:
            mouth_num[2] += 1
        if int(time_find[4:6]) == 1:
            mouth_num[3] += 1
        if int(time_find[4:6]) == 2:
            mouth_num[4] += 1
        if int(time_find[4:6]) == 3:
            mouth_num[5] += 1
        if int(time_find[4:6]) == 4:
            mouth_num[6] += 1

def mouth_ratio_d():
    global mouth_ratio
    global March_to_December
    num = len(mouth_ratio)
    if mouth_num[2] == 0:
        March_to_December = 1
    if mouth_num[2] != 0:
        March_to_December = round((mouth_num[5]*31)/(mouth_num[2]*30),3)
    while(num):
        mouth_ratio[num-1] = round(mouth_num[num-1] / visit_sum,3)
        num -= 1

def P_high_hour():
    global most_P_hour
    global hour_mul
    global most_P_hour_2
    global most_P_hour_num
    global high_hour_hosbital
    global high_hour_eating
    global high_hour_P
    global high_hour_num
    global mutation_hour
    global high_hour_night
    num = 0
    high_h1 = 0
    high_h2 = 0
    high_e1 = 0
    high_e2 = 0
    num_allow = 0
    for i in range(len(hour_mul)):
        if hour[i] != 0:
            hour_mul[i] = round(hour[i+1] / hour[i],3)
        if hour[i] == 0:
            hour_mul[i] = 2
    for i in range(len(hour_mul)-1):
        if hour_mul[i] > 1 and hour_mul[i+1] < 1:
            high_hour_P[num] = (i+1)
            num += 1
        if hour_mul[i] > 2 or hour_mul[i+1] < 0.5:
            mutation_hour = 1
    for i in range(len(high_hour_P)):
        if high_hour_P[i] == 0 and num_allow == 0:
            high_hour_num = i
            num_allow = 1
        if high_hour_P[i] >= 8 and high_hour_P[i] <= 11:
            high_h1 = 1
        if high_hour_P[i] >= 14 and high_hour_P[i] <= 16:
            high_h2 = 1
        if high_hour_P[i] >= 11 and high_hour_P[i] <= 14:
            high_e1 = 1
        if high_hour_P[i] >= 17 and high_hour_P[i] <= 20:
            high_e2 = 1
        if high_hour_P[i] >= 22:
            high_hour_night = 1
    if high_h1 == 1 and high_h2 == 1:
        high_hour_hosbital = 1
    if high_e1 == 1 and high_e2 == 1:
        high_hour_eating = 1

findtxt(root, ret)
for path in ret:
    train_timenum += 1
    if train_timenum >= txt_start:
        with open(path, "a", encoding="utf-8")as f:
            f = open(path, "r", encoding="utf-8")
            for line in f:
                line = line.strip()
                line_list = line.split("	")
                line_date_list = line_list[1].split(",")
                if len(line_date_list) < 100:
                    day[len(line_date_list)] += 1
                time_long(line, line_date_list)
                visit_num(line_date_list)
                day_num(line_date_list)
                time_num(line_date_list)
                march_num(line_date_list)
                visit_ave(line_date_list)
                more_than3day(line_date_list)
                January_num(line_date_list)
                spring_trip(line_date_list)
                hour_long(line_date_list)
                February_num(line_date_list)
                day_ratio(line_date_list)
                mouth_num_d(line_date_list)
                peron_sum += 1
            visit_morethan2_ratio = round(visit_morethan2/peron_sum,3)
            visit_morethan10_ratio = round(visit_morethan10 / peron_sum,3)
            visit_morethan30_ratio = round(visit_morethan30 / peron_sum,3)
            ave_time_daily = round(time_sum / visit_sum,3)
            time_lessthan2_ratio = round(time_lessthan2 / visit_sum,3)
            time_morethan7_ratio = round(time_morethan7 / visit_sum,3)
            time_after_18_ratio = round(time_after_18 / visit_sum, 3)
            ave_visit_P = round(visit_sum / peron_sum, 3)
            weekday1 = round(weekday1_num / visit_sum, 3)
            weekday2 = round(weekday2_num / visit_sum, 3)
            weekday3 = round(weekday3_num / visit_sum, 3)
            weekday4 = round(weekday4_num / visit_sum, 3)
            weekday5 = round(weekday5_num / visit_sum, 3)
            weekday6 = round(weekday6_num / visit_sum, 3)
            weekday7 = round(weekday7_num / visit_sum, 3)
            holiday_num_ratio = round(holiday_num / visit_sum, 3)
            ave_early_hour = round(early_hour_sum / visit_sum, 3)
            ave_leave_hour = round(leave_hour_sum / visit_sum, 3)
            if P_March_num > 0:
                morethan7_P_March_ratio = round(morethan7_P_March_num / P_March_num, 3)
            if P_January_num > 0:
                morethan3_P_January_ratio = round(morethan3_P_January_num / P_January_num, 3)
            morethan3day_P_ratio = round(morethan3day_P_num / visit_sum, 3)
            if visit_sum_2019 > 0:
                spring_trip_P_ratio = round(spring_trip_P_num / visit_sum_2019, 3)
            P_6to8_ratio = round(P_6to8_num / visit_sum, 3)
            P_8to18_ratio = round(P_8to18_num / visit_sum, 3)
            P_visithour14_ratio = round(P_visithour14_num / visit_sum, 3)
            P_visit_more_ratio = round(P_visit_more_num / visit_sum, 3)
            if ave_timegap_num > 0:
                ave_timegap = ave_timegap / ave_timegap_num
            for i in range(len(hour)):
                hour_ratio[i] = hour[i] / hour_sum
            work_8to17 = hour[8] + hour[9] + hour[10] + hour[11] + hour[12] + hour[13] + hour[14] + hour[15] + hour[16] + hour[17]
            relax_18to23 = hour[18] + hour[19] + hour[20] + hour[21] + hour[22] + hour[23]
            if relax_18to23 != 0:
                work_to_relax = work_8to17 / relax_18to23
            most_multiple_day()
            most_multiple_hour()
            if (spring_num[0] + spring_num[1] + spring_num[2]) != 0:
                spring_ratio = round((spring_num[3] + spring_num[4] + spring_num[5]) / (spring_num[0] + spring_num[1] + spring_num[2]),3)
            if (October_num[7] + October_num[8] + October_num[9] + October_num[10] + October_num[11]) != 0:
                Nationalday_weekday_ratio = round(October_num[0] * 5 / (October_num[7] + October_num[8] + October_num[9] + October_num[10] + October_num[11]),3)
            if (October_num[12] + October_num[13]) != 0:
                Nationalday_weekend_ratio = round(October_num[0] * 2 / (October_num[12] + October_num[13]),3)
            if (October_num[4] + October_num[5] + October_num[6]) != 0:
                Nationalday_ratio = round((October_num[0] + October_num[1] + October_num[2]) / (October_num[4] + October_num[5] + October_num[6]),3)
            if November_week_num[1] != 0:
                week_start_to_finish_ratio = round(November_week_num[0] / November_week_num[1],3)
            if NewYearday_num[1] != 0:
                NewYearday_ratio = round(NewYearday_num[0] / NewYearday_num[1],3)
            mouth_ratio_d()
            P_high_hour()
            if weekday_N_num != 0:
                weekend_to_weekday = round((weekend_N_num*5)/(weekday_N_num*2),3)
            if day[2] != 0:
                day30_ratio =round((day[28]+day[29]+day[30])/day[2],3)
            day25to30 = day[25]+day[26]+day[27]+day[28]+day[29]+day[30]
            id = str(path[path.rfind("\\") + 1:path.rfind("_")])
            label = str(path[path.rfind("_") + 1:path.rfind(".")])
            list += [[id, label, peron_sum,visit_sum,ave_visit_P,visit_morethan2_ratio,visit_morethan10_ratio,visit_morethan30_ratio,
                      same_P_mostday,ave_time_daily,time_lessthan2_ratio,time_morethan7_ratio,time_after_18_ratio,
                      weekday1,weekday2,weekday3,weekday4,weekday5,weekday6,weekday7,holiday_num_ratio,most_P_hour,ave_early_hour,
                      ave_leave_hour,morethan7_P_March_ratio,ave_timegap,morethan3_P_January_ratio,morethan3day_P_num,morethan3day_P_ratio,
                      morethan48hour_P_num,morethan72hour_P_num,spring_trip_P_ratio,P_6to8_ratio,P_8to18_ratio,P_visithour14_ratio,P_visit_more_num,
                      P_visit_more_ratio, most_multiple_February, less_multiple_February,  hour_ratio[0], hour_ratio[1], hour_ratio[2], hour_ratio[3],
                              hour_ratio[4], hour_ratio[5], hour_ratio[6],
                              hour_ratio[7], hour_ratio[8], hour_ratio[9], hour_ratio[10], hour_ratio[11], hour_ratio[12],
                              hour_ratio[13], hour_ratio[14], hour_ratio[15],
                              hour_ratio[16], hour_ratio[17], hour_ratio[18], hour_ratio[19], hour_ratio[20], hour_ratio[21],
                              hour_ratio[22], hour_ratio[23], work_to_relax,most_multiple_7to10,most_multiple_11to13,most_multiple_13to15,
                              most_multiple_17to20,less_multiple_7to10,less_multiple_11to13,less_multiple_13to15,less_multiple_17to20,spring_ratio,
                              Nationalday_weekday_ratio,Nationalday_weekend_ratio,Nationalday_ratio,week_start_to_finish_ratio,NewYearday_ratio,
                              mouth_ratio[0],mouth_ratio[1],mouth_ratio[2],mouth_ratio[3],mouth_ratio[4],mouth_ratio[5],March_to_December,high_hour_hosbital,high_hour_eating,high_hour_num,mutation_hour,high_hour_night,weekend_to_weekday,day30_ratio,day25to30]]
            print(train_timenum)
            peron_sum = 0
            visit_sum = 0
            visit_morethan2 = 0
            visit_morethan10 = 0
            visit_morethan30 = 0
            time_sum = 0
            time_lessthan2 = 0
            time_morethan7 = 0
            time_after_18 = 0
            weekday1 = 0
            weekday2 = 0
            weekday3 = 0
            weekday4 = 0
            weekday5 = 0
            weekday6 = 0
            weekday7 = 0
            weekday1_num = 0
            weekday2_num = 0
            weekday3_num = 0
            weekday4_num = 0
            weekday5_num = 0
            weekday6_num = 0
            weekday7_num = 0
            holiday_num = 0
            hour = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            hour_ratio = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            hour_sum = 0
            most_P_hour = 0
            most_P_hour_num = 0
            ave_early_hour = 0
            early_hour_sum = 0
            ave_leave_hour = 0
            leave_hour_sum = 0
            morethan7_P_March_num = 0
            ave_timegap_num = 0
            ave_timegap = 0
            P_March_num = 0
            morethan3_P_January_num = 0
            P_January_num = 0
            morethan3day_P_num = 0
            morethan3day_P_ratio = 0
            morethan48hour_P_num = 0
            morethan72hour_P_num = 0
            visit_sum_2019 = 0
            spring_trip_P_num = 0
            spring_trip_P_ratio = 0
            P_6to8_num = 0
            P_6to8_ratio = 0
            P_8to18_num = 0
            P_8to18_ratio = 0
            P_visithour14_num = 0
            P_visithour14_ratio = 0
            P_visit_more_num = 0
            P_visit_more_ratio = 0
            same_P_mostday = 0
            multiple_February = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            February_people = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            most_multiple_February = 0
            less_multiple_February = 1
            work_to_relax = 0
            work_8to17 = 0
            relax_18to23 = 0
            multiple_hour = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            most_multiple_7to10 = 0
            less_multiple_7to10 = 1
            most_multiple_11to13 = 0
            less_multiple_11to13 = 1
            most_multiple_13to15 = 0
            less_multiple_13to15 = 1
            most_multiple_17to20 = 0
            less_multiple_17to20 = 1
            spring_num = [0, 0, 0, 0, 0, 0]
            spring_ratio = 0
            Nationalday_weekday_ratio = 0
            Nationalday_weekend_ratio = 0
            Nationalday_ratio = 0
            October_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            week_start_to_finish_ratio = 0
            November_week_num = [0, 0]
            NewYearday_ratio = 0
            NewYearday_num = [0, 0]
            mouth_num = [0, 0, 0, 0, 0, 0]
            mouth_ratio = [0, 0, 0, 0, 0, 0]
            March_to_December = 0
            high_hour_hosbital = 0
            high_hour_eating = 0
            most_P_hour_2 = 0
            hour_mul = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            high_hour_P = [0,0,0,0,0,0,0,0,0,0]
            high_hour_num = 0
            mutation_hour = 0
            high_hour_night = 0
            weekend_to_weekday = 1
            weekday_N_num = 0
            weekend_N_num = 0
            day = [0] * 100
            day30_ratio = 0
            day25to30 = 0
            #break
    if train_timenum >= txt_stop:
        write_csv()
        break




