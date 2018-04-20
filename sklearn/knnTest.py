#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
import os
import csv
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as neighbors
import sklearn


path_train = "train.csv"  # 训练文件
path_test = "test.csv"  # 测试文件
# path_train = "/data/dm/train.csv"  # 训练文件
# path_test = "/data/dm/test.csv"  # 测试文件
lowSpeed = 2
lowDirection = 40
highSpeed = 60
lowHeight = 10 #下坡阈值
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。
def ownGroupSpeedLowCount(*arrs,**args2):
    re = 0
    for arr in arrs[0]:
        if (arr < lowSpeed and arr > 0):
            re = re+1
    return int(re)

def ownGroupZeroCount(*arrs,**args2):
    re = 0
    pstate = True
    for arr in arrs[0]:
        if (arr == 0 and  not pstate):
            re = re+1
            pstate = True
        else:
            pstate = False
    return int(re)

def ownGroupCallCount(*arrs,**args2):
    re = 0
    for arr in arrs[0]:
        if (arr > 0  and arr < 4):
            re = re+1
    return int(re)
def ownGroupDirectionCount(*arrs):
    re = 0
    pre = -100
    for value in arrs[0]:
        if(value < 0):
            continue
        if(pre == -100):
            pre = value
            continue
        if(abs(value - pre) > lowDirection):
            re = re + 1
    return int(re)
def ownHignCount(*arrs):
    re = 0
    for arr in arrs[0]:
        if (arr > highSpeed):
            re = re + 1
    return int(re)

def ownHeightLowChange(*arrs):
    re = 0
    pre = -100
    for value in arrs[0]:
        if(pre == -100):
            pre = value
            continue
        if(abs(value - pre) > lowHeight):
            re = re + 1
    return int(re)

def ownY(*arrs):
    for value in arrs[0]:
        return int(value)

def trainData():
    """
    文件读取模块，头文件见columns.
    :return:
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train,sep=',',index_col=None)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    # print(tempdata)
    # 仍然使用自带的iris数据
    # X = tempdata.iloc[0:][['CALLSTATE','SPEED']]
    X = getModel2(tempdata)
    # y = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).Y.agg(ownY).to_frame()['Y'].astype(str).values
    y = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).Y.mean().to_frame()['Y'].astype(int).values
    # print(lowCounts)
    # print(zeroCounts)
    # print(phoneCounts)
    # print(direcctionCounts)

    # knn = neighbors(n_neighbors=5, algorithm='ball_tree').fit(X,y)


    print("------------start pre------------\n")
    tempdata2 = pd.read_csv(path_test,sep=',',index_col=None)
    X2 = getModel2(tempdata2)
    # 预测
    knn = neighbors(5,'distance')
    knn.fit(X,y)
    result = knn.predict(X2)
    print(result)
    # result = pd.concat([tempdata2.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).TERMINALNO.first().to_frame('TERMINALNO'),pd.DataFrame(result)],axis=1)
    # print(result)
    process(tempdata2.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).TERMINALNO.first().to_frame('TERMINALNO'),result)


    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("hh.pdf")
    # Image(graph.create_png())


def process(tempdata,result):
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    import numpy as np

    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
            i = 0
            for indexs in tempdata.index:
                # 此处使用随机值模拟程序预测结果
                writer.writerow([int(tempdata.loc[indexs].values[0:1]), result[i]])  # 随机值
                i=i+1

def getModel(tempdata):
    lowCounts = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).SPEED.agg(
        ownGroupSpeedLowCount).to_frame("lowCount")
    zeroCounts = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).SPEED.agg(
        ownGroupZeroCount).to_frame("zeroCount")
    phoneCounts = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).CALLSTATE.agg(
        ownGroupCallCount).to_frame("callCount")
    direcctionCounts = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).DIRECTION.agg(
        ownGroupDirectionCount).to_frame("directChange")
    X = pd.concat([lowCounts, pd.concat([zeroCounts, pd.concat([phoneCounts, direcctionCounts], axis=1)], axis=1)],
                  axis=1)
    return X

def getModel2(tempdata):
    speed = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).SPEED.agg({'lowCount':ownGroupSpeedLowCount,'zeroCount':ownGroupZeroCount,'highCount':ownHignCount})
    phoneCounts = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).CALLSTATE.agg(
        ownGroupCallCount).to_frame("callCount")
    direcctionCounts = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).DIRECTION.agg(
        ownGroupDirectionCount).to_frame("directChange")
    height = tempdata.sort_values(by=["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).HEIGHT.agg(
        ownHeightLowChange).to_frame("heighChangeCount").astype(int)
    X = pd.concat([speed.astype(int), pd.concat([height, pd.concat([phoneCounts, direcctionCounts], axis=1)], axis=1)],
                  axis=1)
    return X
if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    trainData()
