import operator
from math import log

def splitDataSet(dataSet,axis,value):
    """
    按照给定特征划分数据集
    :param axis:划分数据集的特征的维度
    :param value:特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """

    # 循环遍历dataSet中的每一行数据
    retDataSet = []
    # 找寻 axis下某个特征的非空子集
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis] # 删除这一维特征
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 计算的始终是类别标签的不确定度
def calcShannonEnt(dataSet):
    """
    计算训练数据集中的Y随机变量的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet) # 实例的个数
    labelCounts = {}
    for featVec in dataSet: # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1] # 表示最后一列
        # 当前标签不在labelCounts map中，就让labelCounts加入该标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] =0
        labelCounts[currentLabel] +=1


    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2) # log base 2
    return shannonEnt


def calcConditionalEntropy(dataSet,i,featList,uniqueVals):
    """
    计算x_i给定的条件下，Y的条件熵
    :param dataSet: 数据集
    :param i: 维度i
    :param featList: 数据集特征列表
    :param unqiueVals: 数据集特征集合
    :return: 条件熵
    """
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet,i,value)
        prob = len(subDataSet) / float(len(dataSet)) # 极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet) #∑pH(Y|X=xi) 条件熵的计算
    return ce

# 计算信息增益
def calcInformationGain(dataSet,baseEntropy,i):
    """
    计算信息增益
    :param dataSet: 数据集
    :param baseEntropy: 数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet | X_i)
    """
    featList = [example[i] for example in dataSet] # 第i维特征列表
    uniqueVals = set(featList) # 换成集合 - 集合中的每个元素不重复
    newEntropy = calcConditionalEntropy(dataSet,i,featList,uniqueVals)
    infoGain = baseEntropy - newEntropy # 信息增益
    return infoGain



def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
    :param classList: 类列表
    :retrun: 出现次数最多的类名称
    """

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def chooseBestFeatureToSplitByID3(dataSet):
    """
    选择最好的数据集划分
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) -1 # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): # 遍历所有维度特征
        infoGain = calcInformationGain(dataSet,baseEntropy,i)
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature # 返回最佳特征对应的维度


def calcInformationGainRate(dataSet,baseEntropy,i):
        """
        计算信息增益比
        :param dataSet: 数据集
        :param baseEntropy: 数据集中Y的信息熵
        :param i: 特征维度i
        :return: 特征i对数据集的信息增益g(dataSet|X_i)
        """
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
                currentLabel = featVec[i]
                if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] =0
                labelCounts[currentLabel] +=1
        shannonEnt = 0.0
        for key in labelCounts:
                prob = float(labelCounts[key]) / numEntries
                shannonEnt -= prob * log(prob,2)

        return calcInformationGain(dataSet,baseEntropy,i) / shannonEnt

def chooseBestFeatureToSplitByC45(dataSet):
        """
        选择最好的数据集划分方式
        :param dataSet:
        :return:
        """
        numFeatures = len(dataSet[0]) -1 # 最后一列是分类
        baseEntropy = calcShannonEnt(dataSet)
        bestInfoGainRate =0.0
        bestFeature = -1
        for i in range(numFeatures):
                infoGainRate = calcInformationGainRate(dataSet,baseEntropy,i)
                if (infoGainRate > bestInfoGainRate):
                        bestInfoGainRate = infoGainRate
                        bestFeature = i
        return bestFeature


def noInformationGainToSplitByID3(dataSet):
        """
        不使用信息增益概念，而是直接判断条件熵的大小
        """
        numFeatures = len(dataSet[0]) -1
        bestConditionEntropy = 1.0
        bestFeature =-1
        for i in range(numFeatures):
                featList = [example[i] for example in dataSet]
                uniqueVals = set(featList)
                conditionEntropy = calcConditionalEntropy(dataSet,i,featList,uniqueVals)
                if (conditionEntropy < bestConditionEntropy):
                        bestConditionEntropy = conditionEntropy
                        bestFeature =i
        return bestFeature


def createTree(dataSet,labels,chooseBestFeatureToSplitFunc = chooseBestFeatureToSplitByID3):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 数据集每一维的名称
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet] # 类别列表
    if classList.count(classList[0]) == len(classList): # 统计属于列别classList[0]的个数
        return classList[0] # 当类别完全相同则停止继续划分
    if len(dataSet[0]) ==1: # 当只有一个特征的时候，遍历所有实例返回出现次数最多的类别
        return majorityCnt(classList) # 返回类别标签
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree ={bestFeatLabel:{}}  # map 结构，且key为featureLabel
    del (labels[bestFeat])
    # 找到需要分类的特征子集
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# change map feature to label's index
def mapFeatureToLabelIndex(map,labels):
        for key in map.keys():
                for i in range(len(labels)):
                        if key == labels[i]:
                                return key,i


# 决策树预测函数
def predict(testData,decisionTree,labels):
        # 获得决策树结点的下标
        feature_label,feature_index = mapFeatureToLabelIndex(decisionTree,labels)
        tree = decisionTree[feature_label][testData[feature_index]]
        # 判断该树是叶子结点还是子结点
        if (~isinstance(tree,dict)): # 如果是叶子结点，则直接返回结果
                return tree
        else: # 子结点则继续递归
                return predict(testData,tree,labels)

# 决策树准确率判断
def calPrecision(dataSet,predictSet):
        length = len(dataSet)
        count = 0
        for i in range(length):
              if dataSet[i][-1] == predictSet[i]:
                        count +=1
        return count / length *100

# 测试不同数据集的决策树构建
import loadData as ld
dataSet,labels = ld.createDataSet('car.data')
import copy
predict_labels = copy.copy(labels)

# split dataSet into trainingSet and testSet
import numpy as np
np.random.shuffle(dataSet)

m = len(dataSet)
# 定义交叉验证比例
rate = 0.7
training_len = int(rate * m);

trainingSet,testSet = dataSet[0:training_len],dataSet[training_len:-1]
myTree = createTree(trainingSet,labels,chooseBestFeatureToSplitByC45)

# 预测训练集
predict_result =[]
for data in testSet:
        result = predict(data[0:-1],myTree,predict_labels)
        predict_result.append(result)

# 测试训练集准确率
print("decision Tree predict precision: %.2f"%calPrecision(testSet,predict_result),"%")