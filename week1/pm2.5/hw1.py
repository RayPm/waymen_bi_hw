# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 作业
# 通过LSTM预预测未来北京PM2.5的值
class TsaPred(object):
    def __init__(self):
        self.dataset = None
        self.values = None
        pass

    # 获取数据
    def init_data(self, csv_path, show=False):
        # 展示时候把date做为特征进行展示会特别慢，  index_col = 0设置时间date为序
        self.dataset = pd.read_csv(csv_path, index_col=0)
        self.values = self.dataset.values
        if show:
            print(self.dataset)
            # 展示8个特征
            i = 1
            for group in range(8):
                # 8行1列，第i个
                plt.subplot(8, 1, i)
                # 原始特征的values中的group列进行可视化
                plt.plot(self.values[:, group])
                plt.title(self.dataset.columns[group])
                i += 1
            plt.show()

    # 数据预处理,
    def pretreatment(self):
        v = self.dataset['wnd_dir'].value_counts()
        # wnd_dir 是SE,NW等字符串信息，不适合后期特征计算，需要对其进行标签编码
        encoder = LabelEncoder()
        self.values[:, 4] = encoder.fit_transform(self.values[:, 4])
        self.values = self.values.astype('float32')

    # 数据规范化，正态分布，01规范化等
    def normalize(self):
        # 01规范化
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(self.values)
        return scaled

    # 将数据转换为适合监督学习的数据
    def transform(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 预测序列 (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # 拼接到一起
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # 去掉NaN行
        if dropnan:
            agg.dropna(inplace=True)
        return agg


if __name__ == '__main__':
    tsa = TsaPred()
    tsa.init_data('./pollution.csv')
    tsa.pretreatment()
    scaled = tsa.normalize()
    reframed = tsa.transform(scaled, 1, 1)
    # 去掉不需要预测的列, 只保留需要预测的列
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    # 由于时间序列不连续，所以不能使用shuffle=True的train_test_split方法
    n_train_hours = int(len(reframed.values) * 0.8)
    train = reframed.values[:n_train_hours, :]
    test = reframed.values[n_train_hours:, :]
    # 训练集
    train_X = train[:, :-1]
    # 训练集-真实值
    train_y = train[:, -1]
    # 测试集
    test_X = test[:, :-1]
    # 测试集-真实值
    test_y = test[:, -1]

    # 转换为3D格式：[样本数， 时间步，特征]
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

    # 设置网络模型
    model = Sequential()
    # 输入神经元个数
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # Dense全连接层
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # 模型训练
    # batch_size 执行的大小
    # validation_data 与传入值进行判断。 打印中间结果
    # verbose 打印的状态
    result = model.fit(train_X, train_y, epochs=10, batch_size=64, validation_data=(test_X, test_y), shuffle=False)

    # 模型预测
    train_predict = model.predict(train_X)
    test__predict = model.predict(test_X)

    # 绘制训练损失和测试损失
    train_line = result.history['loss']
    test_line = result.history['val_loss']
    plt.plot(train_line, label='train', c='g')
    plt.plot(test_line, label='test', c='r')
    plt.legend()
    plt.show()

    model.summary()

    # 直观显示结果
    plt.plot(reframed.values[:, -1], label='real', c='b')
    plt.plot([x for x in train_predict], label='trainpredict', c='g')
    plt.plot([None for _ in train_predict] + [x for x in test__predict], label='testpredict', c='r')
    plt.legend()
    plt.show()
