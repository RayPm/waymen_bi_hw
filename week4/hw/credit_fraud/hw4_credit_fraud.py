# -*- coding: utf-8 -*-

# 信用卡欺诈分析：
# 数据集：2013年9月份两天时间内的信用卡交易数据
# 284807笔交易，492笔欺诈行为
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# 数据样本包括了28个特征V1，V2，……V28，以及交易时间Time和交易金额Amount
# 因为数据隐私，28个特征值是通过PCA变换得到的结果。
# 需要预测 每笔交易的分类Class，该笔交易是否为欺诈
# Class=0为正常（非欺诈），Class=1代表欺诈



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 提供了一些函数，用来计算真实值与预测值之间的预测误差
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score


# 误差矩阵可视化
# cm 混淆举证值
# 分类值0或1
# cmap：颜色
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix"', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # 修正X轴的显示
    plt.xticks(tick_marks, classes, rotation=0)
    # 修改Y轴的显示
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    # 增加数值的显示
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 模型的准确率和召回率可视化
def plot_precision_recall(precision, recall, thresholds):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision-recall')
    plt.show();


# 显示模型中特征重要性
def feature_importance(clf):
    # 显示模型中特征重要性
    coeffs = clf.coef_
    df_co = pd.DataFrame(np.transpose(abs(coeffs)), columns=["coef_"])
    # 下标设置为Feature Name
    df_co.index = features.columns
    df_co.sort_values("coef_", ascending=True, inplace=True)
    df_co.coef_.plot(kind="barh")
    plt.title("Feature Importance")
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('../data/creditcard.csv')
    # Amount 数值过大，需要对其进行调整，这里使用正态分布规范化方式
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    # 对训练的结果值进行提取
    y = np.array(data.Class.tolist())
    # 维度，去掉无关的Time和结果集class
    features = data.drop(['Time', 'Class'], axis=1)
    # print(data)
    # print(y)

    # 划分训练和预测数据集
    train_x, test_x, train_y, test_y = train_test_split(features, y, test_size=0.1)
    # 选取模型 -- 逻辑回归LR模型
    clf = LogisticRegression()
    # 训练模型
    clf.fit(train_x, train_y)
    # 预测结果
    predict_y = clf.predict(test_x)

    # 计算混淆矩阵，通过混淆矩阵来评定监督学习算法的性能。
    # TP(True Positive): 真实为0，预测也为0
    # FN(False Negative): 真实为0，预测为1
    # FP(False Positive): 真实为1，预测为0
    # TN(True Negative): 真实为1，预测也为1
    cm = confusion_matrix(test_y, predict_y)

    plot_confusion_matrix(cm, classes=[0, 1], normalize=False, title='Confusion matrix"', cmap=plt.cm.Blues)

    # 预测样本的置信分数
    # 在二分类的情况下，分类模型的decision_function返回结果的形状与样本数量相同，
    # 且返回结果的数值表示模型预测样本属于positive正样本的可信度。
    # 二分类情况下classes_中的第一个标签代表是负样本，第二个标签代表正样本。
    y_score = clf.decision_function(test_x)
    print(y_score)
    print(clf.classes_)
    # 计算准确率，召回率，阈值，用于可视化
    precision, recall, thresholds = precision_recall_curve(test_y, y_score)
    plot_precision_recall(precision, recall, thresholds)

    # 区分accuracy_score， accuracy_score是模型的得分，不能告诉你响应值的潜在分布
    # precision_recall_curve是显示模型内分布的准确率得分
    print(len(precision), precision)
    print(np.mean(precision), accuracy_score(test_y, predict_y))

    # 显示模型中特征重要性
    feature_importance(clf)