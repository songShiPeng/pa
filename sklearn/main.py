from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import os
import csv
import pandas as pd
from IPython.display import Image
from sklearn import tree
import pydotplus

path_train = "train.csv"  # 训练文件
path_test = "test.csv"  # 测试文件
lowSpeed = 3
lowDirection = 30
path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。
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
        if(abs(value - pre) > lowDirection):
            re = re + 1
    return re

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
    lowCounts = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).SPEED.agg(ownGroupSpeedLowCount).to_frame()
    zeroCounts = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).SPEED.agg(ownGroupZeroCount).to_frame()
    phoneCounts = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).CALLSTATE.agg(ownGroupCallCount).to_frame()
    direcctionCounts = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).DIRECTION.agg(ownGroupDirectionCount).to_frame()
    y = tempdata.sort_values(by = ["TERMINALNO", "TIME"]).groupby(['TERMINALNO']).Y.mean().to_frame()['Y'].astype(str).values
    # print(lowCounts)
    # print(zeroCounts)
    # print(phoneCounts)
    # print(direcctionCounts)
    X = pd.concat([lowCounts,pd.concat([zeroCounts,pd.concat([phoneCounts,direcctionCounts],axis=1)],axis=1)],axis=1)
    # X = [lowCounts,zeroCounts,phoneCounts,direcctionCounts]
    # y = tempdata['Y'].astype(str)
    print(X)
    # print("y:的值")
    print(y)
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(criterion='gini',max_depth=50)
    # 拟合模型
    clf.fit(X, y)

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("hh.pdf")
    Image(graph.create_png())


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    import numpy as np

    with open(path_test) as lines:
        with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
            writer = csv.writer(outer)
            i = 0
            ret_set = set([])
            for line in lines:
                if i == 0:
                    i += 1
                    writer.writerow(["Id", "Pred"])  # 只有两列，一列Id为用户Id，一列Pred为预测结果(请注意大小写)。
                    continue
                item = line.split(",")
                if item[0] in ret_set:
                    continue
                # 此处使用随机值模拟程序预测结果
                writer.writerow([item[0], np.random.rand()])  # 随机值

                ret_set.add(item[0])  # 根据赛题要求，ID必须唯一。输出预测值时请注意去重


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    trainData()
