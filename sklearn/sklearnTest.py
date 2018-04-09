from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy as np

#读取数据
data=[]
labels=[]
#根据text里的数据格式将数据写到list里
with open('C:\Users\cchen\Desktop\sample.txt','r') as f:
    for line in f:
        linelist=line.split(' ')
        data.append([float(el) for el in linelist[:-1]])
        labels.append(linelist[-1].strip())
# print data
# [[1.5, 50.0], [1.5, 60.0], [1.6, 40.0], [1.6, 60.0], [1.7, 60.0], [1.7, 80.0], [1.8, 60.0], [1.8, 90.0], [1.9, 70.0], [1.9, 80.0]]
# print labels
x=np.array(data)
labels=np.array(labels)
# print labels
# ['thin' 'fat' 'thin' 'fat' 'thin' 'fat' 'thin' 'fat' 'thin' 'fat']
y=np.zeros(labels.shape)
# print y
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# print labels=='fat'
# [False  True False  True False  True False  True False  True]
# 这个替换的方法很巧妙，可以一学，利用布尔值来给list赋值。要是我的话就要写个循环了。
y[labels=='fat']=1
# print y
# [ 0.  1.  0.  1.  0.  1.  0.  1.  0.  1.]
#拆分训练数据和测试数据,把20%的当做测试数据，其实我感觉直接分片就可以的，不过这样比较高大上一点
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#使用信息熵作为划分标准，对决策树进行训练
clf=tree.DecisionTreeClassifier(criterion='entropy')
# print clf
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')
clf.fit(x_train,y_train)
#把决策树写入文件
with open(r'C:\Users\cchen\Desktop\tree.dot','w+') as f:
    f=tree.export_graphviz(clf,out_file=f)
# digraph Tree {
# node [shape=box] ;
# 0 [label="X[1] <= 70.0\nentropy = 0.9544\nsamples = 8\nvalue = [3, 5]"] ;
# 1 [label="X[0] <= 1.65\nentropy = 0.971\nsamples = 5\nvalue = [3, 2]"] ;
# 0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
# 2 [label="X[1] <= 55.0\nentropy = 0.9183\nsamples = 3\nvalue = [1, 2]"] ;
# 1 -> 2 ;
# 3 [label="entropy = 0.0\nsamples = 1\nvalue = [1, 0]"] ;
# 2 -> 3 ;
# 4 [label="entropy = 0.0\nsamples = 2\nvalue = [0, 2]"] ;
# 2 -> 4 ;
# 5 [label="entropy = 0.0\nsamples = 2\nvalue = [2, 0]"] ;
# 1 -> 5 ;
# 6 [label="entropy = 0.0\nsamples = 3\nvalue = [0, 3]"] ;
# 0 -> 6 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
# }
#系数反应每个特征值的影响力
# print clf.feature_importances_
# [ 0.3608012  0.6391988],可以看到身高系数影响较大
#测试结果打印
anwser=clf.predict(x_train)
# print x_train
print(anwser)
# [ 1.  0.  1.  0.  1.  0.  1.  0.]
print (y_train)
# [ 1.  0.  1.  0.  1.  0.  1.  0.]
print (np.mean(anwser==y_train))
# 1.0 很准，毕竟用的是训练的数据
#让我们用测试的数据来看看
anwser=clf.predict(x_test)
print (anwser)
# [ 0.  0.]
print( y_test)
# [ 0.  0.]
print( np.mean(anwser==y_test))
# 1.0 也很准
#这个是教程里的注释，我没碰到
#准确率与召回率 #准确率：某个类别在测试结果中被正确测试的比率 #召回率：某个类别在真实结果中被正确预测的比率 #测试结果：array([ 0., 1., 0., 1., 0., 1., 0., 1., 0., 0.]) #真实结果：array([ 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]) #分为thin的准确率为0.83。是因为分类器分出了6个thin，其中正确的有5个，因此分为thin的准确率为5/6=0.83。 #分为thin的召回率为1.00。是因为数据集中共有5个thin，而分类器把他们都分对了（虽然把一个fat分成了thin！），召回率5/5=1。 #分为fat的准确率为1.00。不再赘述。 #分为fat的召回率为0.80。是因为数据集中共有5个fat，而分类器只分出了4个（把一个fat分成了thin！），召回率4/5=0.80。 #本例中，目标是尽可能保证找出来的胖子是真胖子（准确率），还是保证尽可能找到更多的胖子（召回率）。
precision,recall,thresholds=precision_recall_curve(y_train,clf.predict(x_train))
print( precision,recall,thresholds)
# [ 1.  1.] [ 1.  0.] [ 1.]
anwser=clf.predict_proba(x)[:,1]
print( classification_report(y,anwser,target_names=['thin','fat']))
#              precision    recall  f1-score   support

       # thin       1.00      1.00      1.00         5
       #  fat       1.00      1.00      1.00         5
#
#  avg / total       1.00      1.00      1.00        10