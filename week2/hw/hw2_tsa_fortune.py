# -*- coding: utf-8 -*-

'''
Action2：资金流入流出预测

数据集一共包括4张表：用户基本信息数据、用户申购赎回数据、收益率表和银行间拆借利率表
2.8万用户，284万行为数据，294天拆解利率，427天收益率
2013-07-01到2014-08-31，预测2014年9月的申购和赎回
'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import copy
import util


if __name__ == '__main__':
    pf = pd.read_csv('../data/user_balance_table.csv', parse_dates=['report_date'])
    total_balance = pf.groupby(['report_date'])[['total_purchase_amt', 'total_redeem_amt']].sum()

    purchase = total_balance[['total_purchase_amt']]
    redeem = total_balance[['total_redeem_amt']]

    '''展示0401-0430的数据情况，用于观察是否平稳
    purchase2 = purchase[(purchase.index >= '2014-04-01') & (purchase.index <= '2014-04-30')]
    plt.figure(figsize=(20, 6))
    plt.plot(purchase2.total_purchase_amt)

    dr = pd.date_range('2014-04-01', '2014-04-30')
    plt.xticks(dr, rotation=90)
    plt.show()
    total_balance.plot(figsize=(20, 6))
    plt.show()
    '''

    # 查看trend（趋势），seasonal（季节性）和residual (残留)
    # util.plot_stl(purchase)


    t = adfuller(purchase)
    target = copy.deepcopy(purchase)
    print(t)
    check = t[4]
    diff_times = 0
    while t:
        # 用概率论判断能否拒绝原假设，说明平不平稳
        if t[0] < check['1%'] and t[0] < check['5%'] and t[0] < check['10%']:
            break
        else:
            diff_times += 1
            diff1 = target.diff(1)
            # 消除NAN
            diff1 = diff1.dropna(axis=0, how='any')
            t = adfuller(diff1)
            target = diff1
    print('purchase， diff_times:', diff_times)

    # 对purchase进行预测
    model = ARIMA(purchase, order=(7, 1, 5)).fit()
    purchase_pred = model.predict('2014-09-01', '2014-09-30', typ='levels')
    print(model.aic)

    # 对redeem进行预测
    model2 = ARIMA(redeem, order=(7, 1, 5)).fit()
    redeem_pred = model2.predict('2014-09-01', '2014-09-30', typ='levels')
    print(model2.aic)

    result = pd.DataFrame()
    result['report_date'] = purchase_pred.index
    result['purchase'] = purchase_pred.values
    result['redeem'] = redeem_pred.values

    # 转换时间格式
    result['report_date'] = result['report_date'].apply(lambda x: str(x).replace('-', '')[:8])
    # 输出purchase和redeem的结果集
    result.to_csv('../result/tsa_fortune_result.csv', header=None, index=None)