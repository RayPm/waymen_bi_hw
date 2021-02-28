import gc
import numpy as np
import pandas as pd

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat,get_feature_names
from deepctr.models import DIN, DIEN, DSIN
from tqdm import tqdm

# deepctr==0.8.4b

from week5.hw.repeat_buyers.RepeatBuyers_xgboost import get_data, matrix_user_info, matrix_user_log, matrix_other

# 获取数据
matrix = get_data()
# 关联user_info
matrix = matrix_user_info(matrix)
# 关联user_log
matrix = matrix_user_log(matrix, True)
# 增加年龄段，购买点击比
matrix = matrix_other(matrix)

# 截取，不缺到定长M个
M=500
for feature in ['hist_merchant_id','hist_action_type']:
    matrix[feature]=matrix[feature].map(lambda x:np.array(x+[0]*(M-len(x)))[:M])

# 分割训练数据和测试数据
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
train_X = train_data.drop(['label'], axis=1)
train_y = np.array(train_data['label'], dtype=float)

train_X['action_type']=3
feature_columns = []
for column in train_X.columns:
  if column != 'hist_merchant_id' and column != 'hist_action_type':
    print(column)
    num = train_X[column].nunique()
    if num > 10000:
        dim = 10
    else:
        if num > 1000:
            dim = 8
        else:
            dim = 4
    print(num)
    if column  == 'user_id':
        feature_columns += [SparseFeat(column, 19111+1, embedding_dim=dim)]
    elif column  == 'merchant_id':
        feature_columns += [SparseFeat(column, 4994+1, embedding_dim=dim)]
    elif column  == 'action_type':
        feature_columns += [SparseFeat(column, 4+1, embedding_dim=dim)]
    else:
        feature_columns += [DenseFeat(column, 1)]

# print(train_X['hist_merchant_id'].shape)
# M = len(train_X['hist_merchant_id'])

print('M=', M)

# maxlen为历史信息的长度，vocabulary_size为onehot的长度
# feature_columns += [VarLenSparseFeat('hist_merchant_id', maxlen=M, vocabulary_size=19111+1, embedding_dim=8, embedding_name='merchant_id'),
#                    VarLenSparseFeat('hist_action_type', maxlen=M, vocabulary_size=4+1, embedding_dim=4, embedding_name='action_type')]


feature_columns += [VarLenSparseFeat(SparseFeat('hist_merchant_id', vocabulary_size=19111+1, embedding_dim=8, embedding_name='merchant_id'), maxlen=M),
                   VarLenSparseFeat(SparseFeat('hist_action_type', vocabulary_size=4+1, embedding_dim=4, embedding_name='action_type'), maxlen=M)]

hist_features=['merchant_id','action_type']
print(feature_columns)

# 使用DIN模型
model=DIN(feature_columns, hist_features)
# 使用Adam优化器，二分类的交叉熵
model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])

# 组装train_model_input，得到feature names，将train_X转换为字典格式
feature_names=list(train_X.columns)
train_model_input = {name:train_X[name].values for name in feature_names}
# histroy输入必须是二维数组
for fea in ['hist_merchant_id','hist_action_type']:
    l = []
    for i in tqdm(train_model_input[fea]):
        l.append(i)
    train_model_input[fea]=np.array(l)
history = model.fit(train_model_input, train_y, verbose=True, epochs=10, validation_split=0.2,batch_size=512)

# 转换test__model_input
test_data['action_type']=3
test_model_input = {name:test_data[name].values for name in feature_names}
from tqdm import tqdm
for fea in ['hist_merchant_id','hist_action_type']:
    l = []
    for i in tqdm(test_model_input[fea]):
        l.append(i)
    test_model_input[fea]=np.array(l)

# 得到预测结果
prob = model.predict(test_model_input)
submission = pd.read_csv('../data_format1/test_format1.csv')
submission['prob'] = pd.Series(prob[:, 1])
submission.to_csv('prediction.csv', index=False)