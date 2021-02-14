# -*- coding: utf-8 -*-

# 信用卡违约率检测
# https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
# 对信用卡使用数据进行建模，预测用户是否下个月产生违约 => 分类问题
# 机器学习算法有很多，比如SVM、决策树、随机森林和KNN => 该使用哪个模型
# 可以使用GridSearchCV工具，找到每个分类器的最优参数和最优分数，最终找到最适合数据集的分类器和此分类器的参数


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb



# Grid Search：一种调参手段；穷举搜索
# 在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果。
# 其原理就像是在数组里找到最大值。
# 这种方法的主要缺点是比较耗时
def GridSearch(pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='accuracy'):
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=model_param_grid, scoring=score)
    search = gridsearch.fit(train_x, train_y)
    print('GridSearch最优参数:', search.best_params_)
    print('GridSearch最优分数:', search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print('准确率 %0.4lf' % accuracy_score(test_y, predict_y))

    response = {}
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y, predict_y)

    return response


if __name__ == '__main__':
    data = pd.read_csv('../data/UCI_Credit_Card.csv')
    next_month = data['default.payment.next.month'].value_counts()
    # 1 表示违约
    # print(next_month)
    # df = pd.DataFrame({'default.payment.next.month': next_month, "values": next_month.values})
    # sns.barplot(x='default.payment.next.month', y='values', data=df)
    # plt.show()
    #
    # 建模,去掉ID字段
    data.drop(['ID'], inplace=True, axis=1)
    target = data['default.payment.next.month'].values
    columns = data.columns.tolist()
    print(columns)
    columns.remove('default.payment.next.month')
    # 特征
    features = data[columns].values
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.3)

    # 构造各种分类器
    classifiers = [
        SVC(),                      # 支持向量机
        DecisionTreeClassifier(),   # 决策树
        RandomForestClassifier(),   # 随机森林
        KNeighborsClassifier()      # KNN
    ]

    # 分类器名称
    classifier_names = [
        'svc',
        'decisiontreeclassifier',
        'randomforestclassifier',
        'kneighborsclassifier'

    ]
    # 分类器参数
    classifier_param_grid = [
        # C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
        # C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
        # svc__gamma:[0.01, 0.05, 0.1] , 最优参数: {'svc__C': 1, 'svc__gamma': 0.05}
        {'svc__C': [1], 'svc__gamma': [0.05]},
        {'decisiontreeclassifier__max_depth': [6, 9, 11]},
        {'randomforestclassifier__n_estimators': [3, 5, 6]},
        {'kneighborsclassifier__n_neighbors': [4, 6, 8]}
    ]

    # for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    #     pipeline = Pipeline([
    #         # 正态分布规范化
    #         ('scaler', StandardScaler()),
    #         # 模型拟合与预测
    #         (model_name, model)
    #     ])
    #     result = GridSearch(pipeline, train_x, train_y, test_x, test_y, model_param_grid, score='accuracy')

    # 使用xgboost和gridsearch对比
    clf1 = xgb.XGBClassifier()
    # 设定网格搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
    xgboost_params = {
        'n_estimators': range(80, 200, 4),
        'max_depth': range(2, 15, 1),
        'learning_rate': np.linspace(0.01, 2, 20),
        'subsample': np.linspace(0.7, 0.9, 20),
        'colsample_bytree': np.linspace(0.5, 0.98, 10),
        'min_child_weight': range(1, 9, 1)
    }
    grid = GridSearchCV(clf1, xgboost_params, cv=3, scoring='neg_log_loss', n_jobs=-1, n_iter=300)
    # 在训练集上训练
    grid.fit(train_x, train_y)
    # 返回最优的训练器
    best_estimator = grid.best_estimator_
    print(best_estimator)
