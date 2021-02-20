# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@file: config.py
@company: FinSight
@author: Zhao Ming
@time: 2018-10-30   21:37:39
"""
import pandas as pd

SUB_DIR = "../submission"

NUM_SPLITS = 3
RANDOM_SEED = 2018

# types of columns of the dataset dataframe

CATEGORICAL_COLS = ['UID', 'day_x', 'mode', 'success', 'time_x', 'os', 'version',
       'device1_x', 'device2_x', 'device_code1_x', 'device_code2_x',
       'device_code3_x', 'mac1_x', 'mac2', 'ip1_x', 'ip2', 'wifi',
       'geo_code_x', 'ip1_sub_x', 'ip2_sub', 'channel', 'day_y', 'time_y',
       'trans_amt', 'amt_src1', 'merchant', 'code1', 'code2', 'trans_type1',
       'acc_id1', 'device_code1_y', 'device_code2_y', 'device_code3_y',
       'device1_y', 'device2_y', 'mac1_y', 'ip1_y', 'bal', 'amt_src2',
       'acc_id2', 'acc_id3', 'geo_code_y', 'trans_type2', 'market_code',
       'market_type', 'ip1_sub_y', 'Tag', 'hour_x', 'minutes_x', 'hour_y',
       'minutes_y', 'day_sub', 'hour_sub', 'minutes_sub', 'amt_bal_sum',
       'geo_code_sub']

NUMERIC_COLS = ["day_x", "day_y","day_sub", "hour_sub", "minutes_sub", "bal", "trans_amt"]

IGNORE_COLS = ["UID", "Tag"]

