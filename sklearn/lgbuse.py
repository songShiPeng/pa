# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from math import radians, cos, sin, asin, sqrt


# path
# path_train = "/data/dm/train.csv"  # 训练文件路径
# path_test = "/data/dm/test.csv"  # 测试文件路径
path_train = "train.csv"  # 训练文件路径
path_test = "test.csv"  # 测试文件路径
path_result_out = "model/pro_result.csv" #预测结果文件路径
lowSpeed = 5
lowDirection = 40
highSpeed = 40
lowHeight = 15 #下坡阈值
def ownGroupSpeedLowCount(*arrs,**args2):
    re = 0
    for arr in arrs[0]:
        if (arr < lowSpeed and arr > 0):
            re = re+1
    return re

def ownGroupZeroCount(*arrs,**args2):
    re = 0
    pstate = True
    for arr in arrs[0]:
        if (arr == 0 and  not pstate):
            re = re+1
            pstate = True
        else:
            pstate = False
    return re

def ownGroupCallCount(*arrs,**args2):
    re = 0
    for arr in arrs[0]:
        if (arr > 0  and arr < 4):
            re = re+1
    return re

def ownGroupDirectionCount(*arrs):
    re = 0
    pre = -100
    for value in arrs[0]:
        if(value < 0):
            continue
        if(pre == -100):
            pre = value
            continue
        if(abs(value - pre) > lowDirection and pre != -100):
            re = re + 1
    return re
def ownHignCount(*arrs):
    re = 0
    for arr in arrs[0]:
        if (arr > highSpeed):
            re = re + 1
    return re

def ownHeightLowChange(*arrs):
    re = 0
    pre = -100
    for value in arrs[0]:
        if(pre == -100):
            pre = value
            continue
        if(abs(value - pre) > lowHeight and pre != -100):
            re = re + 1
    return re
def getModel(data,type):
    train1 = []
    # Feature Engineer, 对每一个用户生成特征:
    # trip特征, record特征(数量,state等),
    # 地理位置特征(location,海拔,经纬度等), 时间特征(星期,小时等), 驾驶行为特征(速度统计特征等)
    for item in data['TERMINALNO'].unique():
        # print('user NO:',item)
        temp = data.loc[data['TERMINALNO'] == item, :]
        # print(temp)
        temp.index = range(len(temp))
        # print(temp)
        # trip 特征
        num_of_trips = temp['TRIP_ID'].nunique()
        # record 特征
        num_of_records = temp.shape[
            0]  # shape函数读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度,相当于行数，建立一个3×3的单位矩阵e, e.shape为（3，3），表示3行3列,第一维的长度为3，第二维的长度也为3
        num_of_state = temp[['TERMINALNO', 'CALLSTATE']]
        # print(num_of_state)
        nsh = num_of_state.shape[0]
        num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE'] == 0].shape[0]
        num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE'] == 1].shape[0]
        num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE'] == 2].shape[0]
        num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE'] == 3].shape[0]
        num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE'] == 4].shape[0]
        # del num_of_state

        ### 地点特征
        startlong = temp.loc[0, 'LONGITUDE']
        startlat = temp.loc[0, 'LATITUDE']
        hdis1 = haversine1(startlong, startlat, 113.9177317, 22.54334333)  # 距离某一点的距离
        # 时间特征
        # 行程时长
        # timeLength = float(temp['TIME'].max()) - float(temp['TIME'].min())
        lonLength = float(temp['LONGITUDE'].max()) - float(temp['LONGITUDE'].min())
        latLength = float(temp['LATITUDE'].max()) - float(temp['LATITUDE'].min())
        zoom = lonLength * latLength
        maxDistance = sqrt(zoom)

        minLong = float(temp['LONGITUDE'].min())
        maxLong = float(temp['LONGITUDE'].max())
        minLat = float(temp['LATITUDE'].min())
        maxLat = float(temp['LATITUDE'].max())
        # temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
        temp['hour'] = temp['TIME'].apply(lambda x: datetime.datetime.fromtimestamp(x).hour)
        hour_state = np.zeros([24, 1])
        for i in range(24):
            hour_state[i] = temp.loc[temp['hour'] == i].shape[0] / float(nsh)
        # 驾驶行为特征
        mean_speed = temp['SPEED'].mean()

        meanTime = maxDistance / mean_speed
        var_speed = temp['SPEED'].var()
        mean_height = temp['HEIGHT'].mean()
        ##次数特征
        lowCounts = temp['SPEED'].agg(
            ownGroupSpeedLowCount)/ float(nsh)
        zeroCounts =temp['SPEED'].agg(
            ownGroupZeroCount)/float(nsh)
        phoneCounts =temp['CALLSTATE'].agg(
            ownGroupCallCount)
        direcctionCounts = temp[["DIRECTION",'TIME']].sort_values(by=["TIME"]).DIRECTION.agg(
            ownGroupDirectionCount)
        # temp2 = temp[['DIRECTION','TIME']]
        highCount = temp['SPEED'].agg(ownHignCount)
        heightCount = temp[["HEIGHT",'TIME']].sort_values(by=["TIME"]).HEIGHT.agg(ownHeightLowChange)
        # 添加label
        if(type == 0):
            target = temp.loc[0, 'Y']
        else:
            target = -1.0
            # 所有特征
        feature = [item, meanTime,zoom, highCount, heightCount, lowCounts, phoneCounts, direcctionCounts,
                   num_of_trips, num_of_records, num_of_state_0, num_of_state_1, num_of_state_2, num_of_state_3,
                   num_of_state_4, \
                   mean_speed, var_speed, mean_height \
            , float(hour_state[0]), float(hour_state[1]), float(hour_state[2]), float(hour_state[3]),
                   float(hour_state[4]), float(hour_state[5])
            , float(hour_state[6]), float(hour_state[7]), float(hour_state[8]), float(hour_state[9]),
                   float(hour_state[10]), float(hour_state[11])
            , float(hour_state[12]), float(hour_state[13]), float(hour_state[14]), float(hour_state[15]),
                   float(hour_state[16]), float(hour_state[17])
            , float(hour_state[18]), float(hour_state[19]), float(hour_state[20]), float(hour_state[21]),
                   float(hour_state[22]), float(hour_state[23])
            , hdis1
            , target]
        train1.append(feature)
    return train1
def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

start_all = datetime.datetime.now()
# read train data
data = pd.read_csv(path_train)

alluser = data['TERMINALNO'].nunique()
train1=[]
train1 = getModel(data,0)
train1 = pd.DataFrame(train1)

# 特征命名
featurename = ['item','meanTime''zoom','highCount','heightCount','lowCounts','phoneCounts','direcctionCounts','num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','num_of_state_3','num_of_state_4',\
              'mean_speed','var_speed','mean_height'
    ,'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11'
    ,'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23'
    ,'dis'
    ,'target']
train1.columns = featurename

print("train data process time:",(datetime.datetime.now()-start_all).seconds)
# 特征使用
feature_use = [ 'zoom','meanTime','highCount','heightCount','lowCounts','phoneCounts','direcctionCounts','num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','num_of_state_3','num_of_state_4',\
               'mean_speed','var_speed','mean_height'
    ,'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11'
    ,'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23'
    ,'dis']


# The same process for the test set
data = pd.read_csv(path_test)
test1 = getModel(data,1)
# make predictions for test data
test1 = pd.DataFrame(test1)
test1.columns = featurename

# 采用lgb回归预测模型，具体参数设置如下
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# 训练、预测
model_lgb.fit(train1[feature_use].fillna(-1), train1['target'])
y_pred = model_lgb.predict(test1[feature_use].fillna(-1))
print("lgb success")

im=model_lgb.feature_importances_
print(im)
print(im.size)

# output result
result = pd.DataFrame(test1['item'])
result['pre'] = y_pred
result = result.rename(columns={'item':'Id','pre':'Pred'})
result.to_csv(path_result_out,header=True,index=False)
print("Time used:",(datetime.datetime.now()-start_all).seconds)
# '''