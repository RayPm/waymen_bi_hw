# -*- coding: utf-8 -*-

'''
Action1：交通流量预测

JetRail高铁的乘客数量预测
数据集：jetrail.csv，根据过往两年的数据（2012 年 8 月至 2014 年 8月），需要用这些数据预测接下来 7 个月的乘客数量
以每天为单位聚合数据集
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet


df = pd.read_csv('../data/train.csv')
# 调整训练集格式
# df.Datetime = pd.to_datetime(df.Datetime)
df.Datetime = pd.to_datetime(df.Datetime, format='%d-%m-%Y %H:%M')
df.index = df.Datetime
df = df.drop(['ID','Datetime'], axis=1)
# 合并日期数据，按日统计
df_day = df.resample('D').sum()
df_day.head()
# 修改列名，采用prophet的保留字作为列名
# df_day.rename(columns={'Datetime':'ds','Count':'y'},inplace=True)
df_day['ds'] = df_day.index
df_day['y'] = df_day['Count']
df_day = df_day.drop(['Count'], axis=1)

# 拟合prophet模型, 预测未来7个月（213天）
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
model.fit(df_day)
future = model.make_future_dataframe(periods=213)
forecast = model.predict(future)
# print(forecast)
# pycharm 不能正常显示，jupyter可以
# model.plot

plt.scatter(df_day['ds'].values, df_day['y'].values)
plt.plot(forecast['ds'],forecast[['trend', 'yhat_lower', 'yhat_upper', 'yhat']])
plt.show()