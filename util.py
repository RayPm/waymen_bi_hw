import matplotlib.pyplot as plt
import statsmodels.api as sm


'''
    通过statsmodels工具
    返回三个部分 trend（趋势），seasonal（季节性）和residual (残留)
'''
def plot_stl(data, isShow=True):
    result = sm.tsa.seasonal_decompose(data, period=30)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    result.trend.plot(ax=ax1, title='Trend')
    result.seasonal.plot(ax=ax2, title='Seasonal')
    result.resid.plot(ax=ax3, title='Residual')

    if isShow:
        plt.show()