import gc
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


def get_data():
    # 训练集,方便做特征工程(user_id 和 label)
    train_data1 = pd.read_csv('../data_format1/train_format1.csv')
    # 测试集
    submission = pd.read_csv('../data_format1/test_format1.csv')

    train_data1['origin'] = 'train'
    submission['origin'] = 'test'
    matrix = pd.concat([train_data1, submission], ignore_index=True, sort=False)

    # 去除需要预测的字段
    matrix.drop(['prob'], axis=1, inplace=True)

    # 手动释放内存
    del train_data1
    gc.collect()
    return matrix


def matrix_user_info(data):
    # 用户画像
    user_info = pd.read_csv('../data_format1/user_info_format1.csv')

    # 连接user_info表，通过user_id关联
    data = data.merge(user_info, on='user_id', how='left')
    data['age_range'].fillna(0, inplace=True)
    # 0:female, 1:male, 2:unknown
    data['gender'].fillna(2, inplace=True)
    data['age_range'] = data['age_range'].astype('int8')
    data['gender'] = data['gender'].astype('int8')
    data['label'] = data['label'].astype('str')
    data['user_id'] = data['user_id'].astype('int32')
    data['merchant_id'] = data['merchant_id'].astype('int32')

    # 手动释放内存
    del user_info
    gc.collect()

    return data


# 关联user_log数据，增加特征值
def matrix_user_log(data):
    # 负采样数据集
    train_data = pd.read_csv('../data_format2/train_format2.csv')
    # 用户行为日志
    user_log = pd.read_csv('../data_format1/user_log_format1.csv', dtype={'time_stamp': 'str'})
    # 使用merchant_id（原列名seller_id）
    user_log.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
    # 格式化
    user_log['user_id'] = user_log['user_id'].astype('int32')
    user_log['merchant_id'] = user_log['merchant_id'].astype('int32')
    user_log['item_id'] = user_log['item_id'].astype('int32')
    user_log['cat_id'] = user_log['cat_id'].astype('int32')
    user_log['brand_id'].fillna(0, inplace=True)
    user_log['brand_id'] = user_log['brand_id'].astype('int32')
    user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%H%M')

    # User特征处理
    groups = user_log.groupby(['user_id'])

    # 用户交互行为数量 u1
    temp = groups.size().reset_index().rename(columns={0: 'u1'})
    data = data.merge(temp, on='user_id', how='left')
    # 商品数量 u2
    temp = groups['item_id'].agg([('u2', 'nunique')]).reset_index()
    data = data.merge(temp, on='user_id', how='left')
    # 所属品类 u3
    temp = groups['cat_id'].agg([('u3', 'nunique')]).reset_index()
    data = data.merge(temp, on='user_id', how='left')
    # 商家数量u4
    temp = groups['merchant_id'].agg([('u4', 'nunique')]).reset_index()
    data = data.merge(temp, on='user_id', how='left')
    # 品牌数量 u5
    temp = groups['brand_id'].agg([('u5', 'nunique')]).reset_index()
    data = data.merge(temp, on='user_id', how='left')

    # 计算时间间隔 u6
    temp = groups['time_stamp'].agg([('F_time', 'min'), ('L_time', 'max')]).reset_index()
    temp['u6'] = (temp['L_time'] - temp['F_time']).dt.seconds / 3600
    data = data.merge(temp[['user_id', 'u6']], on='user_id', how='left')
    # 0表示单击，1表示添加到购物车，2表示购买，3表示添加到收藏夹
    # u7 ,u8, u9, u10
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(
        columns={0: 'u7', 1: 'u8', 2: 'u9', 3: 'u10'})
    data = data.merge(temp, on='user_id', how='left')

    # 商家特征处理
    groups = user_log.groupby(['merchant_id'])
    # 商家被交互行为数量 m1
    temp = groups.size().reset_index().rename(columns={0: 'm1'})
    data = data.merge(temp, on='merchant_id', how='left')
    # 统计商家被交互的user_id, item_id, cat_id, brand_id 唯一值
    temp = groups['user_id', 'item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(
        columns={'user_id': 'm2', 'item_id': 'm3', 'cat_id': 'm4', 'brand_id': 'm5'})
    data = data.merge(temp, on='merchant_id', how='left')
    # 统计商家被交互的action_type 唯一值
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(
        columns={0: 'm6', 1: 'm7', 2: 'm8', 3: 'm9'})
    data = data.merge(temp, on='merchant_id', how='left')
    # 按照merchant_id 统计随机负采样的个数
    temp = train_data[train_data['label'] == -1].groupby(['merchant_id']).size().reset_index().rename(
        columns={0: 'm10'})
    data = data.merge(temp, on='merchant_id', how='left')

    # 按照user_id, merchant_id分组
    groups = user_log.groupby(['user_id', 'merchant_id'])
    # 统计行为个数
    temp = groups.size().reset_index().rename(columns={0: 'um1'})
    data = data.merge(temp, on=['user_id', 'merchant_id'], how='left')
    # 统计item_id, cat_id, brand_id唯一个数
    temp = groups['item_id', 'cat_id', 'brand_id'].nunique().reset_index().rename(
        columns={'item_id': 'um2', 'cat_id': 'um3', 'brand_id': 'um4'})
    data = data.merge(temp, on=['user_id', 'merchant_id'], how='left')
    temp = groups['action_type'].value_counts().unstack().reset_index().rename(
        columns={0: 'um5', 1: 'um6', 2: 'um7', 3: 'um8'})  # 统计不同action_type唯一个数
    data = data.merge(temp, on=['user_id', 'merchant_id'], how='left')
    temp = groups['time_stamp'].agg([('first', 'min'), ('last', 'max')]).reset_index()
    temp['um9'] = (temp['last'] - temp['first']).dt.seconds / 3600
    temp.drop(['first', 'last'], axis=1, inplace=True)
    data = data.merge(temp, on=['user_id', 'merchant_id'], how='left')  # 统计时间间隔

    # 释放内存
    del temp, user_log
    gc.collect()

    return data


# 增加年龄段，购买点击比
def matrix_other(data):
    # 用户购买点击比
    data['r1'] = data['u9'] / data['u7']
    # 商家购买点击比
    data['r2'] = data['m8'] / data['m6']
    # 不同用户不同商家购买点击比
    data['r3'] = data['um7'] / data['um5']
    data.fillna(0, inplace=True)
    # # 修改age_range字段名称为 age_0, age_1, age_2... age_8
    temp = pd.get_dummies(data['age_range'], prefix='age')
    data = pd.concat([data, temp], axis=1)
    temp = pd.get_dummies(data['gender'], prefix='g')
    data = pd.concat([data, temp], axis=1)
    data.drop(['age_range', 'gender'], axis=1, inplace=True)

    return data


# 获取数据
matrix = get_data()
# 关联user_info
matrix = matrix_user_info(matrix)
# 关联user_log
matrix = matrix_user_log(matrix)
# 增加年龄段，购买点击比
matrix = matrix_other(matrix)

# 分割训练数据和测试数据
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['label', 'origin'], axis=1)
train_X, train_y = train_data.drop(['label'], axis=1), train_data['label']

# 将训练集进行切分，20%用于验证
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=.2)

# 使用XGBoost
model = xgb.XGBClassifier(
    max_depth=8,    # 6
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42
)
model.fit(
    X_train, y_train,
    eval_metric='auc', eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True,
    # 早停法，如果auc在10epoch没有进步就stop
    early_stopping_rounds=10
)

model.fit(X_train, y_train)

prob = model.predict_proba(test_data)

submission = pd.read_csv('../data_format1/test_format1.csv')
submission['prob'] = pd.Series(prob[:, 1])
submission.to_csv('prediction.csv', index=False)
