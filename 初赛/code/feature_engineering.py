# encoding:utf-8
import pandas as pd

# 读入操作数据
operation_train = pd.read_csv('data/operation_train_new.csv')
operation_round = pd.read_csv('data/operation_round1_new.csv')
# 按照用户和操作日期排序
operation_train.sort_values(by=['UID', 'day', 'time'], inplace=True)
# 去重
operation_train.drop_duplicates(keep='last', inplace=True)
# 重新索引
operation_train.reset_index(drop=True, inplace=True)
operation_round.sort_values(by=['UID', 'day', 'time'], inplace=True)
operation_round.drop_duplicates(keep='last', inplace=True)
operation_round.reset_index(drop=True, inplace=True)
# 测试集id增加40000,与训练集错开
operation_round['UID'] = operation_round['UID'].apply(lambda x: x+40000)
# 合并数据
operation = pd.concat([operation_train, operation_round], axis=0)
operation.reset_index(drop=True, inplace=True)
# 读入用户及其标签数据
user_train = pd.read_csv('data/tag_train_new.csv')
user_test = pd.read_csv('data/sub.csv')
user_test['UID'] = user_test['UID'].apply(lambda x: x+40000)
user = pd.concat([user_train, user_test], axis=0)
user.reset_index(drop=True, inplace=True)
# 读入交易数据
transaction_train = pd.read_csv('data/transaction_train_new.csv')
transaction_round = pd.read_csv('data/transaction_round1_new.csv')
# 按照用户和操作日期排序
transaction_train.sort_values(by=['UID', 'day', 'time'], inplace=True)
# 去掉重复数据
transaction_train.drop_duplicates(keep='last', inplace=True)
# 重新索引
transaction_train.reset_index(drop=True, inplace=True)
transaction_round.sort_values(by=['UID', 'day', 'time'], inplace=True)
transaction_round.drop_duplicates(keep='last', inplace=True)
transaction_round.reset_index(drop=True, inplace=True)
# id统一增加40000
transaction_round['UID'] = transaction_round['UID'].apply(lambda x: x+40000)
transaction = pd.concat([transaction_train, transaction_round], axis=0)
transaction.reset_index(drop=True, inplace=True)

##############################################
# 操作特征 ###################################
##############################################

# 用户操作总次数
t = operation[['UID']]
t['user_opera_number'] = 1
t1 = t.groupby('UID').agg('sum').reset_index()

# 用户操作总天数
t = operation[['UID', 'day']]
t.rename(columns={'day': 'user_opera_day_number'}, inplace=True)
t2 = t.groupby('UID')['user_opera_day_number'].nunique().reset_index()

# 用户某天操作最多次数
t = operation[['UID', 'day']]
t['user_opera_max_in_oneday'] = 1
t = t.groupby(['UID', 'day']).agg('count').reset_index()
t = t[['UID', 'user_opera_max_in_oneday']]
t138 = t.groupby(['UID']).agg('max').reset_index()

# 用户操作类型总数/计数
t = operation[['UID', 'mode']]
t3 = t.groupby('UID')['mode'].nunique().reset_index()
t3.rename(columns={'mode': 'user_opera_mode_nunique'}, inplace=True)
t4 = t.groupby('UID')['mode'].count().reset_index()
t4.rename(columns={'mode': 'user_opera_mode_count'}, inplace=True)

# 用户操作类型类别特征
t = operation[['UID', 'mode']]
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t139 = t.groupby('UID').agg('sum').reset_index()

# 用户操作状态总数/计数
t = operation[['UID', 'success']]
t5 = t.groupby('UID')['success'].nunique().reset_index()
t5.rename(columns={'success': 'user_opera_success_nunique'}, inplace=True)
t6 = t.groupby('UID')['success'].count().reset_index()
t6.rename(columns={'success': 'user_opera_success_count'}, inplace=True)

# 不同操作下成功次数统计
t = operation[['UID', 'mode', 'success']]
t = t[t['success'] == 1.0]
t = t.groupby(['UID', 'mode']).sum().reset_index()
t.rename(columns={'success': 'success_number'}, inplace=True)
t = pd.get_dummies(t)
col = t.columns.values.tolist()
for i in col[2:]:
    t[i] = t[i]*t['success_number']
t = t.drop('success_number', axis=1)
t150 = t.groupby('UID').sum().reset_index()
t150 = t150[['UID', 'mode_d25caee90b27fa9b','mode_c8741ce15ceac2a4','mode_2f3e878175e34d9c','mode_072eee5c88d380df','mode_acfaded7e04e7ba0','mode_b501fa4fc58206b9','mode_1c341176507fbd9b','mode_ac63e881c4e19402','mode_bf79b3647c0878eb']]
t150.columns = ['UID','mode_d25caee90b27fa9b_success','mode_c8741ce15ceac2a4_success','mode_2f3e878175e34d9c_success','mode_072eee5c88d380df_success','mode_acfaded7e04e7ba0_success','mode_b501fa4fc58206b9_success','mode_1c341176507fbd9b_success','mode_ac63e881c4e19402_success','mode_bf79b3647c0878eb_success']

# 用户操作时间点总数/计数
t = operation[['UID', 'time']]
t7 = t.groupby('UID')['time'].nunique().reset_index()
t7.rename(columns={'time': 'user_opera_time_nunique'}, inplace=True)
t8 = t.groupby('UID')['time'].count().reset_index()
t8.rename(columns={'time': 'user_opera_time_count'}, inplace=True)

# 操作时间点类别，0-6,7-12,13-18,19-24
def time_to_period(x):
    hour = x.split(':')[0]
    hour = int(hour)
    if 0 < hour <= 3:
        return 'period1'
    elif 3< hour <=6:
        return 'period2'
    elif 6 < hour <= 8:
        return 'period3'
    elif 8 < hour <= 12:
        return 'period4'
    elif 12 < hour <= 15:
        return 'period5'
    elif 15 < hour <= 18:
        return 'period6'
    elif 18 < hour <= 21:
        return 'period7'
    else:
        return 'period8'

t = operation[['UID', 'time']]
t['opera_time_to_period'] = t['time'].apply(lambda x: time_to_period(str(x)))
t = t[['UID', 'opera_time_to_period']]
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t146 = t.groupby('UID').agg('sum').reset_index()


# 操作是否有接近整点，半点的记录
def is_close_hour(s):
    hour, mint, second = s.split(':')
    mint = int(mint)
    if mint == 0 or mint == 59 or mint == 30:
        return 'close'
    else:
        return 'not_close'
t = operation[['UID', 'time']]
t['opera_is_close_hour'] = t['time'].apply(lambda x: is_close_hour(str(x)))
t = t[['UID', 'opera_is_close_hour']]
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t148 = t.groupby('UID').agg('sum').reset_index()

# 用户操作时间小时数总数

# 用户操作系统总数/计数
t = operation[['UID', 'os']]
t9 = t.groupby('UID')['os'].nunique().reset_index()
t9.rename(columns={'os': 'user_opera_os_nunique'}, inplace=True)
t10 = t.groupby('UID')['os'].count().reset_index()
t10.rename(columns={'os': 'user_opera_os_count'}, inplace=True)

# 用户操作客户端版本总数/计数
t = operation[['UID', 'version']]
t11 = t.groupby('UID')['version'].nunique().reset_index()
t11.rename(columns={'version': 'user_opera_version_nunique'}, inplace=True)
t12 = t.groupby('UID')['version'].count().reset_index()
t12.rename(columns={'version': 'user_opera_version_count'}, inplace=True)

# 用户操作设备1总数/计数
t = operation[['UID', 'device1']]
t13 = t.groupby('UID')['device1'].nunique().reset_index()
t13.rename(columns={'device1': 'user_opera_device1_nunique'}, inplace=True)
t14 = t.groupby('UID')['device1'].count().reset_index()
t14.rename(columns={'device1': 'user_opera_device1_count'}, inplace=True)

# 用户操作设备2总数/计数
t = operation[['UID', 'device2']]
t15 = t.groupby('UID')['device2'].nunique().reset_index()
t15.rename(columns={'device2': 'user_opera_device2_nunique'}, inplace=True)
t16 = t.groupby('UID')['device2'].count().reset_index()
t16.rename(columns={'device2': 'user_opera_device2_count'}, inplace=True)

# 用户操作设备标识1总数/计数
t = operation[['UID', 'device_code1']]
t19 = t.groupby('UID')['device_code1'].nunique().reset_index()
t19.rename(columns={'device_code1': 'user_opera_device_code1_nunique'}, inplace=True)
t20 = t.groupby('UID')['device_code1'].count().reset_index()
t20.rename(columns={'device_code1': 'user_opera_device_code1_count'}, inplace=True)

# 用户操作设备标识2总数/计数
t = operation[['UID', 'device_code2']]
t21 = t.groupby('UID')['device_code2'].nunique().reset_index()
t21.rename(columns={'device_code2': 'user_opera_device_code2_nunique'}, inplace=True)
t22 = t.groupby('UID')['device_code2'].count().reset_index()
t22.rename(columns={'device_code2': 'user_opera_device_code2_count'}, inplace=True)

# 用户操作设备标识3总数/计数
t = operation[['UID', 'device_code3']]
t23 = t.groupby('UID')['device_code3'].nunique().reset_index()
t23.rename(columns={'device_code3': 'user_opera_device_code3_nunique'}, inplace=True)
t24 = t.groupby('UID')['device_code3'].count().reset_index()
t24.rename(columns={'device_code3': 'user_opera_device_code3_count'}, inplace=True)

# 用户操作mac1地址总数/计数
t = operation[['UID', 'mac1']]
t25 = t.groupby('UID')['mac1'].nunique().reset_index()
t25.rename(columns={'mac1': 'user_opera_mac1_nunique'}, inplace=True)
t26 = t.groupby('UID')['mac1'].count().reset_index()
t26.rename(columns={'mac1': 'user_opera_mac1_count'}, inplace=True)

# 用户操作mac2地址总数/计数
t = operation[['UID', 'mac2']]
t27 = t.groupby('UID')['mac2'].nunique().reset_index()
t27.rename(columns={'mac2': 'user_opera_mac2_nunique'}, inplace=True)
t28 = t.groupby('UID')['mac2'].count().reset_index()
t28.rename(columns={'mac2': 'user_opera_mac2_count'}, inplace=True)

# 用户操作ip1总数/计数
t = operation[['UID', 'ip1']]
t29 = t.groupby('UID')['ip1'].nunique().reset_index()
t29.rename(columns={'ip1': 'user_opera_ip1_nunique'}, inplace=True)
t30 = t.groupby('UID')['ip1'].count().reset_index()
t30.rename(columns={'ip1': 'user_opera_ip1_count'}, inplace=True)

# 用户操作ip2总数/计数
t = operation[['UID', 'ip2']]
t31 = t.groupby('UID')['ip2'].nunique().reset_index()
t31.rename(columns={'ip2': 'user_opera_ip2_nunique'}, inplace=True)
t32 = t.groupby('UID')['ip2'].count().reset_index()
t32.rename(columns={'ip2': 'user_opera_ip2_count'}, inplace=True)

# 用户操作wifi总数/计数
t = operation[['UID', 'wifi']]
t33 = t.groupby('UID')['wifi'].nunique().reset_index()
t33.rename(columns={'wifi': 'user_opera_wifi_nunique'}, inplace=True)
t34 = t.groupby('UID')['wifi'].count().reset_index()
t34.rename(columns={'wifi': 'user_opera_wifi_count'}, inplace=True)

# 用户操作地理位置总数/计数
t = operation[['UID', 'geo_code']]
t35 = t.groupby('UID')['geo_code'].nunique().reset_index()
t35.rename(columns={'geo_code': 'user_opera_geo_code_nunique'}, inplace=True)
t36 = t.groupby('UID')['geo_code'].count().reset_index()
t36.rename(columns={'geo_code': 'user_opera_geo_code_count'}, inplace=True)

# 用户操作ip1_sub总数/计数
t = operation[['UID', 'ip1_sub']]
t37 = t.groupby('UID')['ip1_sub'].nunique().reset_index()
t37.rename(columns={'ip1_sub': 'user_opera_ip1_sub_nunique'}, inplace=True)
t38 = t.groupby('UID')['ip1_sub'].count().reset_index()
t38.rename(columns={'ip1_sub': 'user_opera_ip1_sub_count'}, inplace=True)

# 用户操作ip2_sub总数/计数
t = operation[['UID', 'ip2_sub']]
t39 = t.groupby('UID')['ip2_sub'].nunique().reset_index()
t39.rename(columns={'ip2_sub': 'user_opera_ip2_sub_nunique'}, inplace=True)
t40 = t.groupby('UID')['ip2_sub'].count().reset_index()
t40.rename(columns={'ip2_sub': 'user_opera_ip2_sub_count'}, inplace=True)

###################################################
# 交易特征
###################################################
# 用户总交易次数
t = transaction[['UID']]
t['user_trans_number'] = 1
t41 = t.groupby('UID').agg('sum').reset_index()

# 用户交易平台总数/计数
t = transaction[['UID', 'channel']]
t42 = t.groupby('UID')['channel'].nunique().reset_index()
t42.rename(columns={'channel': 'user_trans_channel_nunique'}, inplace=True)
t43 = t.groupby('UID')['channel'].count().reset_index()
t43.rename(columns={'channel': 'user_trans_channel_count'}, inplace=True)

# 是否在某个平台上进行交易
t = transaction[['UID', 'channel']]
t['channel'] = t['channel'].astype(str)
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t142 = t.groupby('UID').agg('sum').reset_index()

# 用户交易日期总数/计数/最大值/最小值/...
t = transaction[['UID', 'day']]
t44 = t.groupby('UID')['day'].nunique().reset_index()
t44.rename(columns={'day': 'user_trans_day_nunique'}, inplace=True)
t45 = t.groupby('UID')['day'].count().reset_index()
t45.rename(columns={'day': 'user_trans_day_count'}, inplace=True)
t46 = t.groupby('UID')['day'].max().reset_index()
t46.rename(columns={'day': 'user_trans_day_max'}, inplace=True)
t47 = t.groupby('UID')['day'].min().reset_index()
t47.rename(columns={'day': 'user_trans_day_min'}, inplace=True)
t48 = t.groupby('UID')['day'].sum().reset_index()
t48.rename(columns={'day': 'user_trans_day_sum'}, inplace=True)
t49 = t.groupby('UID')['day'].mean().reset_index()
t49.rename(columns={'day': 'user_trans_day_mean'}, inplace=True)
t50 = t.groupby('UID')['day'].std().reset_index()
t50.rename(columns={'day': 'user_trans_day_std'}, inplace=True)
t51 = t.groupby('UID')['day'].median().reset_index()
t51.rename(columns={'day': 'user_trans_day_median'}, inplace=True)
t52 = t.groupby('UID')['day'].var().reset_index()
t52.rename(columns={'day': 'user_trans_day_var'}, inplace=True)
t53 = t.groupby('UID')['day'].skew().reset_index()
t53.rename(columns={'day': 'user_trans_day_skew'}, inplace=True)

# 用户某天交易最多次数
t = transaction[['UID', 'day']]
t['user_trans_max_in_oneday'] = 1
t = t.groupby(['UID', 'day']).agg('count').reset_index()
t = t[['UID', 'user_trans_max_in_oneday']]
t140 = t.groupby(['UID']).agg('max').reset_index()

# 用户交易时间点总数/计数
t = transaction[['UID', 'time']]
t54 = t.groupby('UID')['time'].nunique().reset_index()
t54.rename(columns={'time': 'user_trans_time_nunique'}, inplace=True)
t55 = t.groupby('UID')['time'].count().reset_index()
t55.rename(columns={'time': 'user_trans_time_count'}, inplace=True)

# 交易时间点类别
t = transaction[['UID', 'time']]
t['trans_time_to_period'] = t['time'].apply(lambda x: time_to_period(str(x)))
t = t[['UID', 'trans_time_to_period']]
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t147 = t.groupby('UID').agg('sum').reset_index()

# 用户交易金额总数/计数
t = transaction[['UID', 'trans_amt']]
t56 = t.groupby('UID')['trans_amt'].nunique().reset_index()
t56.rename(columns={'trans_amt': 'user_trans_trans_amt_nunique'}, inplace=True)
t57 = t.groupby('UID')['trans_amt'].count().reset_index()
t57.rename(columns={'trans_amt': 'user_trans_trans_amt_count'}, inplace=True)
t58 = t.groupby('UID')['trans_amt'].max().reset_index()
t58.rename(columns={'trans_amt': 'user_trans_trans_amt_max'}, inplace=True)
t59 = t.groupby('UID')['trans_amt'].min().reset_index()
t59.rename(columns={'trans_amt': 'user_trans_trans_amt_min'}, inplace=True)
t60 = t.groupby('UID')['trans_amt'].sum().reset_index()
t60.rename(columns={'trans_amt': 'user_trans_trans_amt_sum'}, inplace=True)
t61 = t.groupby('UID')['trans_amt'].mean().reset_index()
t61.rename(columns={'trans_amt': 'user_trans_trans_amt_mean'}, inplace=True)
t62 = t.groupby('UID')['trans_amt'].std().reset_index()
t62.rename(columns={'trans_amt': 'user_trans_trans_amt_std'}, inplace=True)
t63 = t.groupby('UID')['trans_amt'].median().reset_index()
t63.rename(columns={'trans_amt': 'user_trans_trans_amt_median'}, inplace=True)
t64 = t.groupby('UID')['trans_amt'].var().reset_index()
t64.rename(columns={'trans_amt': 'user_trans_trans_amt_var'}, inplace=True)
t65 = t.groupby('UID')['trans_amt'].skew().reset_index()
t65.rename(columns={'trans_amt': 'user_trans_trans_amt_skew'}, inplace=True)

# 用户交易资金类型总数/计数
t = transaction[['UID', 'amt_src1']]
t68 = t.groupby('UID')['amt_src1'].nunique().reset_index()
t68.rename(columns={'amt_src1': 'user_trans_amt_src1_nunique'}, inplace=True)
t69 = t.groupby('UID')['amt_src1'].count().reset_index()
t69.rename(columns={'amt_src1': 'user_trans_amt_src1_count'}, inplace=True)

# 交易资金类型类别
t = transaction[['UID', 'amt_src1']]
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t143 = t.groupby('UID').agg('sum').reset_index()

t = transaction[['UID', 'amt_src2']]
t70 = t.groupby('UID')['amt_src2'].nunique().reset_index()
t70.rename(columns={'amt_src2': 'user_trans_amt_src2_nunique'}, inplace=True)
t71 = t.groupby('UID')['amt_src2'].count().reset_index()
t71.rename(columns={'amt_src2': 'user_trans_amt_src2_count'}, inplace=True)

# 用户交易商户标识总数/计数
t = transaction[['UID', 'merchant']]
t74 = t.groupby('UID')['merchant'].nunique().reset_index()
t74.rename(columns={'merchant': 'user_trans_merchant_nunique'}, inplace=True)
t75 = t.groupby('UID')['merchant'].count().reset_index()
t75.rename(columns={'merchant': 'user_trans_merchant_count'}, inplace=True)

# 用户交易商户标识code1总数/计数
t = transaction[['UID', 'code1']]
t76 = t.groupby('UID')['code1'].nunique().reset_index()
t76.rename(columns={'code1': 'user_trans_code1_nunique'}, inplace=True)
t77 = t.groupby('UID')['code1'].count().reset_index()
t77.rename(columns={'code1': 'user_trans_code1_count'}, inplace=True)

# 用户交易商户标识code2总数/计数
t = transaction[['UID', 'code2']]
t78 = t.groupby('UID')['code2'].nunique().reset_index()
t78.rename(columns={'code2': 'user_trans_code2_nunique'}, inplace=True)
t79 = t.groupby('UID')['code2'].count().reset_index()
t79.rename(columns={'code2': 'user_trans_code2_count'}, inplace=True)

# 用户交易类型总数/计数
t = transaction[['UID', 'trans_type1']]
t80 = t.groupby('UID')['trans_type1'].nunique().reset_index()
t80.rename(columns={'trans_type1': 'user_trans_trans_type1_nunique'}, inplace=True)
t81 = t.groupby('UID')['trans_type1'].count().reset_index()
t81.rename(columns={'trans_type1': 'user_trans_trans_type1_count'}, inplace=True)

# 交易类型1
t = transaction[['UID', 'trans_type1']]
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t141 = t.groupby('UID').agg('sum').reset_index()

# 不同交易类型下交易金额总数
t = transaction[['UID', 'trans_type1', 'trans_amt']]
t.rename(columns={'trans_type1': 'trans_type1_money'}, inplace=True)
t = t.groupby(['UID', 'trans_type1_money'])['trans_amt'].sum().reset_index()
t.reset_index(drop=True, inplace=True)
t = pd.get_dummies(t)
col = t.columns.values.tolist()
for i in col[2:]:
    t[i] = t[i]*t['trans_amt']
t = t.drop('trans_amt', axis=1)
t149 = t.groupby('UID').sum().reset_index()

# 用户交易类型总数/计数
t = transaction[['UID', 'trans_type2']]
t82 = t.groupby('UID')['trans_type2'].nunique().reset_index()
t82.rename(columns={'trans_type2': 'user_trans_trans_type2_nunique'}, inplace=True)
t83 = t.groupby('UID')['trans_type2'].count().reset_index()
t83.rename(columns={'trans_type2': 'user_trans_trans_type2_count'}, inplace=True)
t84 = t.groupby('UID')['trans_type2'].max().reset_index()
t84.rename(columns={'trans_type2': 'user_trans_trans_type2_max'}, inplace=True)
t85 = t.groupby('UID')['trans_type2'].min().reset_index()
t85.rename(columns={'trans_type2': 'user_trans_trans_type2_min'}, inplace=True)
t86 = t.groupby('UID')['trans_type2'].sum().reset_index()
t86.rename(columns={'trans_type2': 'user_trans_trans_type2_sum'}, inplace=True)
t87 = t.groupby('UID')['trans_type2'].mean().reset_index()
t87.rename(columns={'trans_type2': 'user_trans_trans_type2_mean'}, inplace=True)
t88 = t.groupby('UID')['trans_type2'].median().reset_index()
t88.rename(columns={'trans_type2': 'user_trans_trans_type2_median'}, inplace=True)
t89 = t.groupby('UID')['trans_type2'].std().reset_index()
t89.rename(columns={'trans_type2': 'user_trans_trans_type2_std'}, inplace=True)
t90 = t.groupby('UID')['trans_type2'].var().reset_index()
t90.rename(columns={'trans_type2': 'user_trans_trans_type2_var'}, inplace=True)
t91 = t.groupby('UID')['trans_type2'].skew().reset_index()
t91.rename(columns={'trans_type2': 'user_trans_trans_type2_skew'}, inplace=True)

# 交易类型2
t = transaction[['UID', 'trans_type2']]
t['trans_type2'] = t['trans_type2'].astype(str)
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t144 = t.groupby('UID').agg('sum').reset_index()
t144.fillna(0, inplace=True)
t144['trans_type2_102.0/trans_type2_103.0'] = t144.apply(lambda x: (x['trans_type2_102.0']+1) / (x['trans_type2_103.0']+1), axis=1)
t144['trans_type2_105.0/trans_type2_104.0'] = t144.apply(lambda x: (x['trans_type2_105.0']+1) / (x['trans_type2_104.0']+1), axis=1)

# 用户交易账户相关总数/计数
t = transaction[['UID', 'acc_id1']]
t92 = t.groupby('UID')['acc_id1'].nunique().reset_index()
t92.rename(columns={'acc_id1': 'user_trans_acc_id1_nunique'}, inplace=True)
t93 = t.groupby('UID')['acc_id1'].count().reset_index()
t93.rename(columns={'acc_id1': 'user_trans_acc_id1_count'}, inplace=True)

# 用户交易账户相关总数/计数
t = transaction[['UID', 'acc_id2']]
t94 = t.groupby('UID')['acc_id2'].nunique().reset_index()
t94.rename(columns={'acc_id2': 'user_trans_acc_id2_nunique'}, inplace=True)
t95 = t.groupby('UID')['acc_id2'].count().reset_index()
t95.rename(columns={'acc_id2': 'user_trans_acc_id2_count'}, inplace=True)

# 用户交易账户相关总数/计数
t = transaction[['UID', 'acc_id3']]
t96 = t.groupby('UID')['acc_id3'].nunique().reset_index()
t96.rename(columns={'acc_id3': 'user_trans_acc_id3_nunique'}, inplace=True)
t97 = t.groupby('UID')['acc_id3'].count().reset_index()
t97.rename(columns={'acc_id3': 'user_trans_acc_id3_count'}, inplace=True)

# 用户交易设备总数/计数
t = transaction[['UID', 'device_code1']]
t98 = t.groupby('UID')['device_code1'].nunique().reset_index()
t98.rename(columns={'device_code1': 'user_trans_device_code1_nunique'}, inplace=True)
t99 = t.groupby('UID')['device_code1'].count().reset_index()
t99.rename(columns={'device_code1': 'user_trans_device_code1_count'}, inplace=True)

# 用户交易设备总数/计数
t = transaction[['UID', 'device_code2']]
t100 = t.groupby('UID')['device_code2'].nunique().reset_index()
t100.rename(columns={'device_code2': 'user_trans_device_code2_nunique'}, inplace=True)
t101 = t.groupby('UID')['device_code2'].count().reset_index()
t101.rename(columns={'device_code2': 'user_trans_device_code2_count'}, inplace=True)

# 用户交易设备总数/计数
t = transaction[['UID', 'device_code3']]
t102 = t.groupby('UID')['device_code3'].nunique().reset_index()
t102.rename(columns={'device_code3': 'user_trans_device_code3_nunique'}, inplace=True)
t103 = t.groupby('UID')['device_code3'].count().reset_index()
t103.rename(columns={'device_code3': 'user_trans_device_code3_count'}, inplace=True)

# 用户交易设备总数/计数
t = transaction[['UID', 'device1']]
t104 = t.groupby('UID')['device1'].nunique().reset_index()
t104.rename(columns={'device1': 'user_trans_device1_nunique'}, inplace=True)
t105 = t.groupby('UID')['device1'].count().reset_index()
t105.rename(columns={'device1': 'user_trans_device1_count'}, inplace=True)

# 用户交易设备总数/计数
t = transaction[['UID', 'device2']]
t106 = t.groupby('UID')['device2'].nunique().reset_index()
t106.rename(columns={'device2': 'user_trans_device2_nunique'}, inplace=True)
t107 = t.groupby('UID')['device2'].count().reset_index()
t107.rename(columns={'device2': 'user_trans_device2_count'}, inplace=True)

# 用户交易mac地址总数/计数
t = transaction[['UID', 'mac1']]
t108 = t.groupby('UID')['mac1'].nunique().reset_index()
t108.rename(columns={'mac1': 'user_trans_mac1_nunique'}, inplace=True)
t109 = t.groupby('UID')['mac1'].count().reset_index()
t109.rename(columns={'mac1': 'user_trans_mac1_count'}, inplace=True)

# 用户交易ip地址总数/计数
t = transaction[['UID', 'ip1']]
t110 = t.groupby('UID')['ip1'].nunique().reset_index()
t110.rename(columns={'ip1': 'user_trans_ip1_nunique'}, inplace=True)
t111 = t.groupby('UID')['ip1'].count().reset_index()
t111.rename(columns={'ip1': 'user_trans_ip1_count'}, inplace=True)

# 用户交易账户余额总数/计数
t = transaction[['UID', 'bal']]
t112 = t.groupby('UID')['bal'].nunique().reset_index()
t112.rename(columns={'bal': 'user_trans_bal_nunique'}, inplace=True)
t113 = t.groupby('UID')['bal'].count().reset_index()
t113.rename(columns={'bal': 'user_trans_bal_count'}, inplace=True)
t114 = t.groupby('UID')['bal'].max().reset_index()
t114.rename(columns={'bal': 'user_trans_bal_max'}, inplace=True)
t115 = t.groupby('UID')['bal'].min().reset_index()
t115.rename(columns={'bal': 'user_trans_bal_min'}, inplace=True)
t116 = t.groupby('UID')['bal'].sum().reset_index()
t116.rename(columns={'bal': 'user_trans_bal_sum'}, inplace=True)
t117 = t.groupby('UID')['bal'].mean().reset_index()
t117.rename(columns={'bal': 'user_trans_bal_mean'}, inplace=True)
t118 = t.groupby('UID')['bal'].std().reset_index()
t118.rename(columns={'bal': 'user_trans_bal_std'}, inplace=True)
t119 = t.groupby('UID')['bal'].median().reset_index()
t119.rename(columns={'bal': 'user_trans_bal_median'}, inplace=True)
t120 = t.groupby('UID')['bal'].var().reset_index()
t120.rename(columns={'bal': 'user_trans_bal_var'}, inplace=True)
t121 = t.groupby('UID')['bal'].skew().reset_index()
t121.rename(columns={'bal': 'user_trans_bal_skew'}, inplace=True)

# 用户交易地理位置总数/计数
t = transaction[['UID', 'geo_code']]
t122 = t.groupby('UID')['geo_code'].nunique().reset_index()
t122.rename(columns={'geo_code': 'user_trans_geo_code_nunique'}, inplace=True)
t123 = t.groupby('UID')['geo_code'].count().reset_index()
t123.rename(columns={'geo_code': 'user_trans_geo_code_count'}, inplace=True)

# 用户交易营销活动编码总数/计数
t = transaction[['UID', 'market_code']]
t124 = t.groupby('UID')['market_code'].nunique().reset_index()
t124.rename(columns={'market_code': 'user_trans_market_code_nunique'}, inplace=True)
t125 = t.groupby('UID')['market_code'].count().reset_index()
t125.rename(columns={'market_code': 'user_trans_market_code_count'}, inplace=True)

# 用户交易营销活动标识总数/计数
t = transaction[['UID', 'market_type']]
t126 = t.groupby('UID')['market_type'].nunique().reset_index()
t126.rename(columns={'market_type': 'user_trans_market_type_nunique'}, inplace=True)
t127 = t.groupby('UID')['market_type'].count().reset_index()
t127.rename(columns={'market_type': 'user_trans_market_type_count'}, inplace=True)
t128 = t.groupby('UID')['market_type'].max().reset_index()
t128.rename(columns={'market_type': 'user_trans_market_type_max'}, inplace=True)
t129 = t.groupby('UID')['market_type'].min().reset_index()
t129.rename(columns={'market_type': 'user_trans_market_type_min'}, inplace=True)
t130 = t.groupby('UID')['market_type'].sum().reset_index()
t130.rename(columns={'market_type': 'user_trans_market_type_sum'}, inplace=True)
t131 = t.groupby('UID')['market_type'].mean().reset_index()
t131.rename(columns={'market_type': 'user_trans_market_type_mean'}, inplace=True)
t132 = t.groupby('UID')['market_type'].median().reset_index()
t132.rename(columns={'market_type': 'user_trans_market_type_median'}, inplace=True)
t133 = t.groupby('UID')['market_type'].std().reset_index()
t133.rename(columns={'market_type': 'user_trans_market_type_std'}, inplace=True)
t134 = t.groupby('UID')['market_type'].var().reset_index()
t134.rename(columns={'market_type': 'user_trans_market_type_var'}, inplace=True)
t135 = t.groupby('UID')['market_type'].skew().reset_index()
t135.rename(columns={'market_type': 'user_trans_market_type_skew'}, inplace=True)

# 营销活动标识
t = transaction[['UID', 'market_type']]
t['market_type'] = t['market_type'].astype(str)
t = pd.get_dummies(t)
t.reset_index(drop=True, inplace=True)
t145 = t.groupby('UID').agg('sum').reset_index()
t145.fillna(0, inplace=True)
t145['market_type_1.0/market_type_2.0'] = t145.apply(lambda x: (x['market_type_1.0']+1) / (x['market_type_2.0']+1), axis=1)
t145['market_type_2.0/market_type_nan'] = t145.apply(lambda x: (x['market_type_2.0']+1) / (x['market_type_nan']+1), axis=1)

# 用户交易ip子地址总数/计数
t = transaction[['UID', 'ip1_sub']]
t136 = t.groupby('UID')['ip1_sub'].nunique().reset_index()
t136.rename(columns={'ip1_sub': 'user_trans_ip1_sub_nunique'}, inplace=True)
t137 = t.groupby('UID')['ip1_sub'].count().reset_index()
t137.rename(columns={'ip1_sub': 'user_trans_ip1_sub_count'}, inplace=True)

# 用户不同交易类型下的各种值
t = transaction[['UID', 'trans_amt', 'trans_type1']]
t = t[t['trans_type1'] == 'c2f2023d279665b2'][['UID', 'trans_amt']]
t151 = t.groupby('UID').count().reset_index()
t151.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_count'}, inplace=True)
t152 = t.groupby('UID').max().reset_index()
t152.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_max'}, inplace=True)
t153 = t.groupby('UID').min().reset_index()
t153.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_min'}, inplace=True)
t154 = t.groupby('UID').sum().reset_index()
t154.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_sum'}, inplace=True)
t155 = t.groupby('UID').mean().reset_index()
t155.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_mean'}, inplace=True)
t156 = t.groupby('UID').std().reset_index()
t156.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_std'}, inplace=True)
t157 = t.groupby('UID').median().reset_index()
t157.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_median'}, inplace=True)
t158 = t.groupby('UID').var().reset_index()
t158.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_var'}, inplace=True)
t159 = t.groupby('UID').skew().reset_index()
t159.rename(columns={'trans_amt': 'trans_type1_c2f2023d279665b2_skew'}, inplace=True)

t = transaction[['UID', 'trans_amt', 'trans_type1']]
t = t[t['trans_type1'] == '6d55c54c8b1056fb'][['UID', 'trans_amt']]
t160 = t.groupby('UID').count().reset_index()
t160.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_count'}, inplace=True)
t161 = t.groupby('UID').max().reset_index()
t161.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_max'}, inplace=True)
t162 = t.groupby('UID').min().reset_index()
t162.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_min'}, inplace=True)
t163 = t.groupby('UID').sum().reset_index()
t163.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_sum'}, inplace=True)
t164 = t.groupby('UID').mean().reset_index()
t164.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_mean'}, inplace=True)
t165 = t.groupby('UID').std().reset_index()
t165.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_std'}, inplace=True)
t166 = t.groupby('UID').median().reset_index()
t166.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_median'}, inplace=True)
t167 = t.groupby('UID').var().reset_index()
t167.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_var'}, inplace=True)
t168 = t.groupby('UID').skew().reset_index()
t168.rename(columns={'trans_amt': 'trans_type1_6d55c54c8b1056fb_skew'}, inplace=True)

#################################################################################
# 交叉特征 ######################################################################
#################################################################################

# 操作系统版本号大小

# 合并特征
feature = pd.merge(user, t1, how='outer', on='UID')
feature = pd.merge(feature, t2, how='outer', on='UID')
feature = pd.merge(feature, t3, how='outer', on='UID')
feature = pd.merge(feature, t4, how='outer', on='UID')
feature = pd.merge(feature, t5, how='outer', on='UID')
feature = pd.merge(feature, t6, how='outer', on='UID')
feature = pd.merge(feature, t7, how='outer', on='UID')
feature = pd.merge(feature, t8, how='outer', on='UID')
feature = pd.merge(feature, t9, how='outer', on='UID')
feature = pd.merge(feature, t10, how='outer', on='UID')
feature = pd.merge(feature, t11, how='outer', on='UID')
feature = pd.merge(feature, t12, how='outer', on='UID')
feature = pd.merge(feature, t13, how='outer', on='UID')
feature = pd.merge(feature, t14, how='outer', on='UID')
feature = pd.merge(feature, t15, how='outer', on='UID')
feature = pd.merge(feature, t16, how='outer', on='UID')
feature = pd.merge(feature, t19, how='outer', on='UID')
feature = pd.merge(feature, t20, how='outer', on='UID')
feature = pd.merge(feature, t21, how='outer', on='UID')
feature = pd.merge(feature, t22, how='outer', on='UID')
feature = pd.merge(feature, t23, how='outer', on='UID')
feature = pd.merge(feature, t24, how='outer', on='UID')
feature = pd.merge(feature, t25, how='outer', on='UID')
feature = pd.merge(feature, t26, how='outer', on='UID')
feature = pd.merge(feature, t27, how='outer', on='UID')
feature = pd.merge(feature, t28, how='outer', on='UID')
feature = pd.merge(feature, t29, how='outer', on='UID')
feature = pd.merge(feature, t30, how='outer', on='UID')
feature = pd.merge(feature, t31, how='outer', on='UID')
feature = pd.merge(feature, t32, how='outer', on='UID')
feature = pd.merge(feature, t33, how='outer', on='UID')
feature = pd.merge(feature, t34, how='outer', on='UID')
feature = pd.merge(feature, t35, how='outer', on='UID')
feature = pd.merge(feature, t36, how='outer', on='UID')
feature = pd.merge(feature, t37, how='outer', on='UID')
feature = pd.merge(feature, t38, how='outer', on='UID')
feature = pd.merge(feature, t39, how='outer', on='UID')
feature = pd.merge(feature, t40, how='outer', on='UID')
feature = pd.merge(feature, t41, how='outer', on='UID')
feature = pd.merge(feature, t42, how='outer', on='UID')
feature = pd.merge(feature, t43, how='outer', on='UID')
feature = pd.merge(feature, t44, how='outer', on='UID')
feature = pd.merge(feature, t45, how='outer', on='UID')
feature = pd.merge(feature, t46, how='outer', on='UID')
feature = pd.merge(feature, t47, how='outer', on='UID')
feature = pd.merge(feature, t48, how='outer', on='UID')
feature = pd.merge(feature, t49, how='outer', on='UID')
feature = pd.merge(feature, t50, how='outer', on='UID')
feature = pd.merge(feature, t51, how='outer', on='UID')
feature = pd.merge(feature, t52, how='outer', on='UID')
feature = pd.merge(feature, t53, how='outer', on='UID')
feature = pd.merge(feature, t54, how='outer', on='UID')
feature = pd.merge(feature, t55, how='outer', on='UID')
feature = pd.merge(feature, t56, how='outer', on='UID')
feature = pd.merge(feature, t57, how='outer', on='UID')
feature = pd.merge(feature, t58, how='outer', on='UID')
feature = pd.merge(feature, t59, how='outer', on='UID')
feature = pd.merge(feature, t60, how='outer', on='UID')
feature = pd.merge(feature, t61, how='outer', on='UID')
feature = pd.merge(feature, t62, how='outer', on='UID')
feature = pd.merge(feature, t63, how='outer', on='UID')
feature = pd.merge(feature, t64, how='outer', on='UID')
feature = pd.merge(feature, t65, how='outer', on='UID')
feature = pd.merge(feature, t68, how='outer', on='UID')
feature = pd.merge(feature, t69, how='outer', on='UID')
feature = pd.merge(feature, t70, how='outer', on='UID')
feature = pd.merge(feature, t71, how='outer', on='UID')
feature = pd.merge(feature, t74, how='outer', on='UID')
feature = pd.merge(feature, t75, how='outer', on='UID')
feature = pd.merge(feature, t76, how='outer', on='UID')
feature = pd.merge(feature, t77, how='outer', on='UID')
feature = pd.merge(feature, t78, how='outer', on='UID')
feature = pd.merge(feature, t79, how='outer', on='UID')
feature = pd.merge(feature, t80, how='outer', on='UID')
feature = pd.merge(feature, t81, how='outer', on='UID')
feature = pd.merge(feature, t82, how='outer', on='UID')
feature = pd.merge(feature, t83, how='outer', on='UID')
feature = pd.merge(feature, t84, how='outer', on='UID')
feature = pd.merge(feature, t85, how='outer', on='UID')
feature = pd.merge(feature, t86, how='outer', on='UID')
feature = pd.merge(feature, t87, how='outer', on='UID')
feature = pd.merge(feature, t88, how='outer', on='UID')
feature = pd.merge(feature, t89, how='outer', on='UID')
feature = pd.merge(feature, t90, how='outer', on='UID')
feature = pd.merge(feature, t91, how='outer', on='UID')
feature = pd.merge(feature, t92, how='outer', on='UID')
feature = pd.merge(feature, t93, how='outer', on='UID')
feature = pd.merge(feature, t94, how='outer', on='UID')
feature = pd.merge(feature, t95, how='outer', on='UID')
feature = pd.merge(feature, t96, how='outer', on='UID')
feature = pd.merge(feature, t97, how='outer', on='UID')
feature = pd.merge(feature, t98, how='outer', on='UID')
feature = pd.merge(feature, t99, how='outer', on='UID')
feature = pd.merge(feature, t100, how='outer', on='UID')
feature = pd.merge(feature, t101, how='outer', on='UID')
feature = pd.merge(feature, t102, how='outer', on='UID')
feature = pd.merge(feature, t103, how='outer', on='UID')
feature = pd.merge(feature, t104, how='outer', on='UID')
feature = pd.merge(feature, t105, how='outer', on='UID')
feature = pd.merge(feature, t106, how='outer', on='UID')
feature = pd.merge(feature, t107, how='outer', on='UID')
feature = pd.merge(feature, t108, how='outer', on='UID')
feature = pd.merge(feature, t109, how='outer', on='UID')
feature = pd.merge(feature, t110, how='outer', on='UID')
feature = pd.merge(feature, t111, how='outer', on='UID')
feature = pd.merge(feature, t112, how='outer', on='UID')
feature = pd.merge(feature, t113, how='outer', on='UID')
feature = pd.merge(feature, t114, how='outer', on='UID')
feature = pd.merge(feature, t115, how='outer', on='UID')
feature = pd.merge(feature, t116, how='outer', on='UID')
feature = pd.merge(feature, t117, how='outer', on='UID')
feature = pd.merge(feature, t118, how='outer', on='UID')
feature = pd.merge(feature, t119, how='outer', on='UID')
feature = pd.merge(feature, t120, how='outer', on='UID')
feature = pd.merge(feature, t121, how='outer', on='UID')
feature = pd.merge(feature, t122, how='outer', on='UID')
feature = pd.merge(feature, t123, how='outer', on='UID')
feature = pd.merge(feature, t124, how='outer', on='UID')
feature = pd.merge(feature, t125, how='outer', on='UID')
feature = pd.merge(feature, t126, how='outer', on='UID')
feature = pd.merge(feature, t127, how='outer', on='UID')
feature = pd.merge(feature, t128, how='outer', on='UID')
feature = pd.merge(feature, t129, how='outer', on='UID')
feature = pd.merge(feature, t130, how='outer', on='UID')
feature = pd.merge(feature, t131, how='outer', on='UID')
feature = pd.merge(feature, t132, how='outer', on='UID')
feature = pd.merge(feature, t133, how='outer', on='UID')
feature = pd.merge(feature, t134, how='outer', on='UID')
feature = pd.merge(feature, t135, how='outer', on='UID')
feature = pd.merge(feature, t136, how='outer', on='UID')
feature = pd.merge(feature, t137, how='outer', on='UID')
feature = pd.merge(feature, t138, how='outer', on='UID')
feature = pd.merge(feature, t139, how='outer', on='UID')
feature = pd.merge(feature, t140, how='outer', on='UID')
feature = pd.merge(feature, t141, how='outer', on='UID')
feature = pd.merge(feature, t142, how='outer', on='UID')
feature = pd.merge(feature, t143, how='outer', on='UID')
feature = pd.merge(feature, t144, how='outer', on='UID')
feature = pd.merge(feature, t145, how='outer', on='UID')
feature = pd.merge(feature, t146, how='outer', on='UID')
feature = pd.merge(feature, t147, how='outer', on='UID')
feature = pd.merge(feature, t148, how='outer', on='UID')
feature = pd.merge(feature, t149, how='outer', on='UID')
feature = pd.merge(feature, t150, how='outer', on='UID')
feature = pd.merge(feature, t151, how='outer', on='UID')
feature = pd.merge(feature, t152, how='outer', on='UID')
feature = pd.merge(feature, t153, how='outer', on='UID')
feature = pd.merge(feature, t154, how='outer', on='UID')
feature = pd.merge(feature, t155, how='outer', on='UID')
feature = pd.merge(feature, t156, how='outer', on='UID')
feature = pd.merge(feature, t157, how='outer', on='UID')
feature = pd.merge(feature, t158, how='outer', on='UID')
feature = pd.merge(feature, t159, how='outer', on='UID')
feature = pd.merge(feature, t160, how='outer', on='UID')
feature = pd.merge(feature, t161, how='outer', on='UID')
feature = pd.merge(feature, t162, how='outer', on='UID')
feature = pd.merge(feature, t163, how='outer', on='UID')
feature = pd.merge(feature, t164, how='outer', on='UID')
feature = pd.merge(feature, t165, how='outer', on='UID')
feature = pd.merge(feature, t166, how='outer', on='UID')
feature = pd.merge(feature, t167, how='outer', on='UID')
feature = pd.merge(feature, t168, how='outer', on='UID')
# feature.to_csv('org_feature.csv')
# 特征缺失率
feature_name = feature.columns.values.tolist()
feature['loss_rate'] = (feature.shape[1] - feature.count(axis=1))/feature.shape[1]

# 缺失值填充
feature.fillna(feature.mean()[feature_name], inplace=True)

# 不同类型交易金额总数差/比例
feature['trans_type1_c2f2023d279665b2_sum - trans_type1_6d55c54c8b1056fb_sum'] = feature.apply(lambda x: x['trans_type1_c2f2023d279665b2_sum'] - x['trans_type1_6d55c54c8b1056fb_sum'], axis=1)
feature['trans_type1_c2f2023d279665b2_sum/trans_type1_6d55c54c8b1056fb_sum'] = feature.apply(lambda x: x['trans_type1_c2f2023d279665b2_sum'] / (1+x['trans_type1_6d55c54c8b1056fb_sum']), axis=1)

# 操作和交易设备总数
feature['opera_device_code_sum'] = feature.apply(lambda x: x['user_opera_device_code1_nunique'] + x['user_opera_device_code2_nunique'] + x['user_opera_device_code3_nunique'], axis=1)
feature['trans_device_code_sum'] = feature.apply(lambda x: x['user_trans_device_code1_nunique'] + x['user_trans_device_code2_nunique'] + x['user_trans_device_code3_nunique'], axis=1)
# 总操作数除总交易数
feature['opera_num/trans_num'] = feature.apply(lambda x: (x['user_opera_number']+1) / (x['user_trans_number']+1), axis=1)
# 总交易金额除总操作次数
feature['trans_amt_sum/opera_number'] = feature.apply(lambda x: x['user_trans_trans_amt_sum'] / (x['user_opera_number']+1), axis=1)
# 总操作天数除总交易天数
feature['opera_day/trans_day'] = feature.apply(lambda x: (x['user_opera_day_number']+1) /(x['user_trans_day_nunique']+1), axis=1)
# 总操作ip1数除总操作地理位置数
feature['opera_ip1_nunique/opera_geo_code_nunique'] = feature.apply(lambda x: (x['user_opera_ip1_nunique']+1) /(x['user_opera_geo_code_nunique']+1), axis=1)
# 总操作ip2数除总操作地理位置数
feature['opera_ip2_nunique/opera_geo_code_nunique'] = feature.apply(lambda x: (x['user_opera_ip2_nunique']+1) /(x['user_opera_geo_code_nunique']+1), axis=1)
# 总交易数除总交易地理位置数
feature['opera_num/trans_num'] = feature.apply(lambda x: (x['user_opera_number']+1) /(1+x['user_trans_geo_code_nunique']), axis=1)
# 总交易金额除以总设备数
feature['trans_amt_sum/trans_device_code_sum'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) /(1+x['trans_device_code_sum']), axis=1)
# 用户操作ip1.nunique+ip2.nunique
feature['user_opera_ip1_nunique+user_opera_ip2_nunique'] = feature.apply(lambda x: (1+x['user_opera_ip1_nunique']) +(1+x['user_opera_ip2_nunique']), axis=1)
# 用户操作ip1_sub.count+ip2_sub.count
feature['user_opera_ip1_count+user_opera_ip2_count'] = feature.apply(lambda x: (1+x['user_opera_ip1_count']) +(1+x['user_opera_ip2_count']), axis=1)
# 用户操作ip1_sub.nunique+ip2_sub.nunique
feature['user_opera_ip1_sub_nunique+user_opera_ip2_sub_nunique'] = feature.apply(lambda x: x['user_opera_ip1_sub_nunique'] + x['user_opera_ip2_sub_nunique'], axis=1)
# 用户交易金额总数/参加活动数
feature['trans_amt_sum/trans_market_type_count'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_trans_market_type_count']), axis=1)
feature['trans_amt_sum/trans_market_type_nunique'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_trans_market_type_nunique']), axis=1)
# 用户交易金额总数/商铺个数
feature['trans_amt_sum/trans_merchant_nunique'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_trans_market_type_count']), axis=1)
feature['trans_amt_sum/trans_merchant_count'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_trans_market_type_nunique']), axis=1)
# 用户交易金额总数/ wifi数
feature['trans_amt_sum/opera_wifi_count'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_opera_wifi_count']), axis=1)
# 用户交易金额总数/地理位置数
feature['trans_amt_sum/opera_geo_code_count'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_opera_geo_code_count']), axis=1)
# 用户交易金额总数/mac数
feature['trans_amt_sum/opera_mac1_count'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_opera_mac1_count']), axis=1)
feature['trans_amt_sum/trans_mac1_count'] = feature.apply(lambda x: (1+x['user_trans_trans_amt_sum']) / (1+x['user_trans_mac1_count']), axis=1)
# 用户操作设备数/mac数量
feature['opera_device_code_sum/opera_mac1_nunique'] = feature.apply(lambda x: (1+x['opera_device_code_sum']) / (1+x['user_opera_mac1_nunique']), axis=1)
# 用户交易设备数/mac数量
feature['trans_device_code_sum/trans_mac1_nunique'] = feature.apply(lambda x: (1+x['trans_device_code_sum']) / (1+x['user_trans_mac1_nunique']), axis=1)
# 用户操作ip数/mac数量
feature['user_opera_ip1_count/opera_mac1_nunique'] = feature.apply(lambda x: (1+x['user_opera_ip1_count']) / (1+x['user_opera_mac1_nunique']), axis=1)
# 用户不同类型交易金额总数与活动的关系
feature['trans_type1_money_c2f2023d279665b2/user_trans_market_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_trans_market_code_nunique']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_trans_market_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_trans_market_code_nunique']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_trans_market_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_trans_market_code_nunique']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_trans_market_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_trans_market_code_nunique']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_trans_market_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_trans_market_code_nunique']), axis=1)

feature['trans_type1_money_c2f2023d279665b2/user_trans_market_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_trans_market_code_count']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_trans_market_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_trans_market_code_count']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_trans_market_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_trans_market_code_count']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_trans_market_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_trans_market_code_count']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_trans_market_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_trans_market_code_count']), axis=1)
# 不用操作之间比值
feature['trans_type1_money_c2f2023d279665b2/trans_type1_money_6d55c54c8b1056fb'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['trans_type1_money_6d55c54c8b1056fb']), axis=1)
feature['trans_type1_money_c2f2023d279665b2/trans_type1_money_61bfb66c928f36ac'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['trans_type1_money_61bfb66c928f36ac']), axis=1)
feature['trans_type1_money_c2f2023d279665b2/trans_type1_money_26bcf43a19df14c8'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['trans_type1_money_26bcf43a19df14c8']), axis=1)
feature['trans_type1_money_c2f2023d279665b2/trans_type1_money_e0d7b8768da99dd4'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['trans_type1_money_e0d7b8768da99dd4']), axis=1)
feature['trans_type1_money_c2f2023d279665b2/trans_type1_money_d9c417304a5ae70c'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['trans_type1_money_d9c417304a5ae70c']), axis=1)

feature['trans_type1_money_6d55c54c8b1056fb/trans_type1_money_61bfb66c928f36ac'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['trans_type1_money_61bfb66c928f36ac']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/trans_type1_money_26bcf43a19df14c8'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['trans_type1_money_26bcf43a19df14c8']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/trans_type1_money_e0d7b8768da99dd4'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['trans_type1_money_e0d7b8768da99dd4']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/trans_type1_money_d9c417304a5ae70c'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['trans_type1_money_d9c417304a5ae70c']), axis=1)

feature['trans_type1_money_61bfb66c928f36ac/trans_type1_money_26bcf43a19df14c8'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['trans_type1_money_26bcf43a19df14c8']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/trans_type1_money_e0d7b8768da99dd4'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['trans_type1_money_e0d7b8768da99dd4']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/trans_type1_money_d9c417304a5ae70c'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['trans_type1_money_d9c417304a5ae70c']), axis=1)

feature['trans_type1_money_26bcf43a19df14c8/trans_type1_money_e0d7b8768da99dd4'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['trans_type1_money_e0d7b8768da99dd4']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/trans_type1_money_d9c417304a5ae70c'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['trans_type1_money_d9c417304a5ae70c']), axis=1)

feature['trans_type1_money_e0d7b8768da99dd4/trans_type1_money_d9c417304a5ae70c'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['trans_type1_money_d9c417304a5ae70c']), axis=1)
# 用户不同类型交易金额总数与交易地理位置的关系
feature['trans_type1_money_c2f2023d279665b2/user_trans_geo_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_trans_geo_code_nunique']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_trans_geo_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_trans_geo_code_nunique']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_trans_geo_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_trans_geo_code_nunique']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_trans_geo_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_trans_geo_code_nunique']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_trans_geo_code_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_trans_geo_code_nunique']), axis=1)
feature['trans_type1_money_c2f2023d279665b2/user_trans_geo_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_trans_geo_code_count']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_trans_geo_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_trans_geo_code_count']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_trans_geo_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_trans_geo_code_count']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_trans_geo_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_trans_geo_code_count']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_trans_geo_code_count'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_trans_geo_code_count']), axis=1)
# 用户不同类型交易金额总数与操作类型关系
feature['trans_type1_money_c2f2023d279665b2/user_opera_mode_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_opera_mode_nunique']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_opera_mode_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_opera_mode_nunique']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_opera_mode_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_opera_mode_nunique']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_opera_mode_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_opera_mode_nunique']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_opera_mode_nunique'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_opera_mode_nunique']), axis=1)
feature['trans_type1_money_c2f2023d279665b2/user_opera_mode_count'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_opera_mode_count']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_opera_mode_count'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_opera_mode_count']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_opera_mode_count'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_opera_mode_count']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_opera_mode_count'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_opera_mode_count']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_opera_mode_count'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_opera_mode_count']), axis=1)
# 用户不同类型交易金额总数/用户操作总数
feature['trans_type1_money_c2f2023d279665b2/user_opera_number'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_opera_number']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_opera_number'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_opera_number']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_opera_number'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_opera_number']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_opera_number'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_opera_number']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_opera_number'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_opera_number']), axis=1)
# 用户不同类型交易金额总数/总交易金额
feature['trans_type1_money_c2f2023d279665b2/user_trans_trans_amt_sum'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_trans_trans_amt_sum']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_trans_trans_amt_sum'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_trans_trans_amt_sum']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_trans_trans_amt_sum'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_trans_trans_amt_sum']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_trans_trans_amt_sum'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_trans_trans_amt_sum']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_trans_trans_amt_sum'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_trans_trans_amt_sum']), axis=1)
# 用户不同类型交易金额总数/交易的平均天数
feature['trans_type1_money_c2f2023d279665b2/user_trans_day_mean'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_trans_day_mean']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_trans_day_mean'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_trans_day_mean']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_trans_day_mean'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_trans_day_mean']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_trans_day_mean'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_trans_day_mean']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_trans_day_mean'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_trans_day_mean']), axis=1)
# 用户不同类型交易金额总数/操作总天数
feature['trans_type1_money_c2f2023d279665b2/user_opera_day_number'] = feature.apply(lambda x: (1+x['trans_type1_money_c2f2023d279665b2']) / (1+x['user_opera_day_number']), axis=1)
feature['trans_type1_money_6d55c54c8b1056fb/user_opera_day_number'] = feature.apply(lambda x: (1+x['trans_type1_money_6d55c54c8b1056fb']) / (1+x['user_opera_day_number']), axis=1)
feature['trans_type1_money_61bfb66c928f36ac/user_opera_day_number'] = feature.apply(lambda x: (1+x['trans_type1_money_61bfb66c928f36ac']) / (1+x['user_opera_day_number']), axis=1)
feature['trans_type1_money_26bcf43a19df14c8/user_opera_day_number'] = feature.apply(lambda x: (1+x['trans_type1_money_26bcf43a19df14c8']) / (1+x['user_opera_day_number']), axis=1)
feature['trans_type1_money_e0d7b8768da99dd4/user_opera_day_number'] = feature.apply(lambda x: (1+x['trans_type1_money_e0d7b8768da99dd4']) / (1+x['user_opera_day_number']), axis=1)
# 用户不同类型交易/总交易次数
feature['trans_type1_c2f2023d279665b2/user_trans_trans_type1_count'] = feature.apply(lambda x: (1+x['trans_type1_c2f2023d279665b2']) / (1+x['user_trans_trans_type1_count']), axis=1)
feature['trans_type1_6d55c54c8b1056fb/user_trans_trans_type1_count'] = feature.apply(lambda x: (1+x['trans_type1_6d55c54c8b1056fb']) / (1+x['user_trans_trans_type1_count']), axis=1)
feature['trans_type1_61bfb66c928f36ac/user_trans_trans_type1_count'] = feature.apply(lambda x: (1+x['trans_type1_61bfb66c928f36ac']) / (1+x['user_trans_trans_type1_count']), axis=1)
feature['trans_type1_26bcf43a19df14c8/user_trans_trans_type1_count'] = feature.apply(lambda x: (1+x['trans_type1_26bcf43a19df14c8']) / (1+x['user_trans_trans_type1_count']), axis=1)
feature['trans_type1_e0d7b8768da99dd4/user_trans_trans_type1_count'] = feature.apply(lambda x: (1+x['trans_type1_e0d7b8768da99dd4']) / (1+x['user_trans_trans_type1_count']), axis=1)

feature.to_csv('feature.csv')
