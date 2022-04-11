import pandas as pd
import os

# print(f'__file__ : {__file__}')
# cwd = os.getcwd()  # Get the current working directory (cwd)
# # c = os.pardir()
# # print(c)
# pth = os.path.dirname(__file__)
# files = os.listdir(pth)  # Get all the files in that directory
#
# # p = pd.read_csv("../../data/train/output.csv")
# # print(p.head())



# print("Files in %r: %s" % (cwd, files))

# k = pd.read_csv("../../data/train/test2_submission.csv", index_col=[0])
# print('k',k.shape)
# print(k.head)
# sample = pd.read_csv("../../data/eval/sample_submission.csv")
# # k.drop(['Unnamed: 0'], axis='columns', inplace=True)
# print('sample',sample.shape)
# # item_ids = list(k[k['user_id'] == 11].values[0])
# # print(item_ids)


k = pd.read_csv('../data/train/genre_submission.csv')
print(k.head())