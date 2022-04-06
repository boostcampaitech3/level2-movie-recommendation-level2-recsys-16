import pandas as pd
import os

print(f'__file__ : {__file__}')
p = pd.read_csv("../check_test+.csv")
print(p.head())

k = pd.read_csv("../../data/train/test2_submission.csv", index_col=[0])
print('k',k.shape)
print(k.head)
sample = pd.read_csv("../../data/eval/sample_submission.csv")
# k.drop(['Unnamed: 0'], axis='columns', inplace=True)
print('sample',sample.shape)
# item_ids = list(k[k['user_id'] == 11].values[0])
# print(item_ids)