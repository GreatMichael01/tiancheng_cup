# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@file: main.py
@company: FinSight
@author: Zhao Ming
@time: 2018-10-30   21:45:17
"""
import warnings
warnings.filterwarnings("ignore")
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import config
import DataReader
import metrics
from DeepFM import DeepFM


start = time.clock()


def load_data():

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

    test_C["Tag"] = -1
    data = pd.concat([train, test_C])
    data["time_x"] = data["time_x"].fillna(value="00:00:00")
    data["time_y"] = data["time_y"].fillna(value="00:00:00")
    data["day_x"] = data["day_x"].fillna(value=0)
    data["day_y"] = data["day_y"].fillna(value=30)
    data = data.fillna(value=-99999)

    data["hour_x"] = data.time_x.apply(lambda x: x.split(":")[0])
    data["minutes_x"] = data.time_x.apply(lambda x: x.split(":")[1])
    data["hour_y"] = data.time_y.apply(lambda x: x.split(":")[0])
    data["minutes_y"] = data.time_y.apply(lambda x: x.split(":")[1])

    data["day_y"] = data["day_y"] + 30
    data["day_sub"] = data["day_y"] - data["day_x"]
    data["hour_sub"] = (data["hour_y"].astype(int) + 24) - data["hour_x"].astype(int)
    data["minutes_sub"] = (data["minutes_y"].astype(int) + 60) - data["minutes_x"].astype(int)
    data["amt_bal_sum"] = data["trans_amt"] + data["bal"]

    dfTrain = data[data["Tag"] != -1]
    dfTest = data[data["Tag"] == -1]


    def preprocess(df):
        cols = [c for c in df.columns if c not in ['UID', 'Tag']]
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ['UID', 'Tag']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['Tag'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['UID'].values

    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = DataReader.FeatureDictionary(dfTrain=dfTrain,
                                          dfTest=dfTest,
                                          numeric_cols=config.NUMERIC_COLS,
                                          ignore_cols=config.IGNORE_COLS)
    data_parser = DataReader.DataParser(feat_dict=fd)
    # Xi_train ：列的序号
    # Xv_train ：列的对应的值
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    print(dfTrain.dtypes)

    dfm_params['feature_size'] = fd.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)

    _get = lambda x, l: [x[i] for i in l]

    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params['epoch']), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params['epoch']), dtype=float)

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)  # 对验证集进行测试
        y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)  # 对测试集数据测试

        gini_results_cv[i] = metrics.gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)

    # _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    aucplot(y_valid_, y_train_meta[valid_idx, 0], clf_str)
    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")



def aucplot(y, y_predprobs, model_name):
    '''
    '''

    fpr, tpr, threshold = roc_curve(y, y_predprobs)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.5f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(model_name + "ROC")
    plt.legend(loc="lower right")
    if os.path.exists("../submission") == False:
        os.makedirs("../submission")
    plt.savefig("../submission/" + model_name + "_roc_" + str(
        int(time.time())) + ".png")
    plt.show()


dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": metrics.gini_norm,
    "random_seed": config.RANDOM_SEED

}
if __name__ == "__main__":
    # load data
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))

    # y_train_dfm,y_test_dfm = run_base_model_dfm(dfTrain,dfTest,folds,dfm_params)
    y_train_dfm, y_test_dfm = run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

    # ------------------ FM Model ------------------
    # fm_params = dfm_params.copy()
    # fm_params["use_deep"] = False
    # y_train_fm, y_test_fm = run_base_model_dfm(dfTrain, dfTest, folds, fm_params)
    #
    # # ------------------ DNN Model ------------------
    # dnn_params = dfm_params.copy()
    # dnn_params["use_fm"] = False
    # y_train_dnn, y_test_dnn = run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)

    end = time.clock()
    print("time: %s seconds" % (end - start))



