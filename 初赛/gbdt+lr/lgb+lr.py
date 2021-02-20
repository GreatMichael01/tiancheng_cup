# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@file: lgb+lr.py
@company: FinSight
@author: Zhao Ming
@time: 2018-11-01   15:20:10
"""

import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def prepro():
    print("读取数据...")
    operation_TRAIN = pd.read_csv("../data/operation_TRAIN.csv")
    transaction_TRAIN = pd.read_csv("../data/transaction_TRAIN.csv")
    tag_TRAIN = pd.read_csv("../data/tag_TRAIN.csv")
    operation_TRAIN = operation_TRAIN.drop_duplicates("UID", keep="last")
    transaction_TRAIN = transaction_TRAIN.drop_duplicates("UID", keep="last")
    train_X = pd.merge(operation_TRAIN, transaction_TRAIN, on="UID", how="outer")
    train = pd.merge(train_X, tag_TRAIN, on="UID", how="left")
    train = train.sort_values("UID")

    operation_round1 = pd.read_csv("../data/operation_round1.csv")
    transaction_round1 = pd.read_csv("../data/transaction_round1.csv")
    operation_round1 = operation_round1.drop_duplicates(["UID"], keep="last")
    transaction_round1 = transaction_round1.drop_duplicates(["UID"], keep="last")
    test_C = pd.merge(operation_round1, transaction_round1, on="UID", how="outer")
    test_C = test_C.sort_values("UID")
    test_id = test_C.UID

    print("读取结束...")

    test_C["Tag"] = -1
    data = pd.concat([train, test_C])
    data["time_x"] = data["time_x"].fillna(value="00:00:00")
    data["time_y"] = data["time_y"].fillna(value="00:00:00")
    data["day_x"] = data["day_x"].fillna(value=0)
    data["day_y"] = data["day_y"].fillna(value=30)
    data = data.fillna(value=-99999)

    data["hour_x"] = data.time_x.apply(lambda  x: x.split(":")[0])
    data["minutes_x"] = data.time_x.apply(lambda  x: x.split(":")[1])
    data["hour_y"] = data.time_y.apply(lambda  x: x.split(":")[0])
    data["minutes_y"] = data.time_y.apply(lambda  x: x.split(":")[1])

    data["day_y"] = data["day_y"] + 30
    data["day_sub"] = data["day_y"] - data["day_x"]
    data["hour_sub"] = (data["hour_y"].astype(int) +24) - data["hour_x"].astype(int)
    data["minutes_sub"] = (data["minutes_y"].astype(int) +60) - data["minutes_x"].astype(int)

    feature = data.columns.drop(["UID", "Tag"])
    col_feature = ["day_x", "hour_sub", "minutes_sub", "bal", "trans_amt"]
    dis_feature = feature.drop(col_feature)

    return data, col_feature, dis_feature, test_id


def gbdt_lr_predict(data, dis_feature, con_feature, test_id):
    """离散特征one-hot"""
    print("开始one-hot...")
    for col in dis_feature:
        lbl = preprocessing.LabelEncoder()
        data[col] = lbl.fit_transform(list(data[col].values))
    print("one-hot结束")

    train = data[data["Tag"] != -1]
    target = train.pop("Tag")
    test = data[data["Tag"] == -1]
    test.drop(["Tag"], axis=1, inplace=True)

    print("划分数据集...")
    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2018)

    print("开始训练gbdt...")

    # lgb模型
    # lgbm = lgb.LGBMClassifier(booster="gbtree",
    #                         subsample=0.8,
    #                         min_child_weight=0.2,
    #                         colsample_bytree=0.8,
    #                         num_leaves=100,
    #                         max_depth=20,
    #                         reg_lambda=20,
    #                         reg_alpha=10,
    #                         learning_rate=0.01,
    #                         n_estimators=2000,
    #                         n_jobs=-1)
    # lgbm.fit(x_train, y_train,
    #         eval_set=[(x_train, y_train), (x_val, y_val)],
    #         eval_names=["train", "val"],
    #         eval_metric="binary_logloss")
    # model = lgbm.booster_

    params = {"booster": "gbtree",
              "objective": "binary:logistic",
              "eval_metric": "auc",
              "verbose": 1,
              "eta": 0.01,
              "max_delta_step": 5,
              "max_depth": 20,
              "alpha": 10,
              "lambda": 10,
              "gamma": 2,
              "subsample": 0.8,
              "colsample_bytree": 0.8,
              "min_child_weight": 5,
              "scale_pos_weight": 0.8,
              "n_jobs": -1}
    matrix_train = xgb.DMatrix(x_train, y_train)
    matrix_test = xgb.DMatrix(x_val)
    model = xgb.train(params, matrix_train, num_boost_round=50000)
    # model = model.predict(matrix_test)

    print("训练得到叶子数...")
    gbdt_feats_train = model.predict(matrix_train, pred_leaf=True)
    gbdt_feats_test = model.predict(matrix_test, pred_leaf=True)
    gbdt_feats_name = ["gbdt_leaf_" + str(i) for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(gbdt_feats_test, columns=gbdt_feats_name)

    print("构造新的数据集...")
    train = pd.concat([train, df_train_gbdt_feats], axis=1)
    test = pd.concat([test, df_test_gbdt_feats], axis=1)
    train_len = train.shape[0]
    data = pd.concat([train, test])
    del train
    del test
    gc.collect()

    """连续特征归一化"""
    print("开始归一化...")
    scaler = MinMaxScaler()
    for col in con_feature:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    print("归一化结束")

    """叶子数One-Hot"""
    print("开始one-hot...")
    for col in gbdt_feats_name:
        print("this is feature:", col)
        lbl = preprocessing.LabelEncoder()
        data[col] = lbl.fit_transform(list(data[col].values))
    print("one-hot结束")

    train = data[:train_len]
    test = data[train_len:]
    del data
    gc.collect()

    x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.3, random_state=2018)
    print("开始训练lr...")
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
    print("tr-logloss:", tr_logloss)
    val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
    print("val-logloss:", val_logloss)

    fpr, tpr, threshold = roc_curve(y_val, lr.predict_proba(x_val)[:, 1])
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.5f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("gbdt_lr" + "ROC")
    plt.legend(loc="lower right")
    plt.show()

    print("开始预测...")
    y_pred = lr.predict_proba(test)[:, 1]
    print("写入结果...")
    submission = pd.DataFrame({"UID": test_id, "Tag": y_pred})
    submission.to_csv("../submission/"+str(int(time.time()))+".csv", index=False)
    print("结束")


if __name__ == "__main__":
    print("time start:\n")
    print(time.clock() / 60)

    data, col_feature, dis_feature, test_id = prepro()
    gbdt_lr_predict(data, dis_feature, col_feature, test_id)

    print('time end')
    print(time.clock() / 60)



