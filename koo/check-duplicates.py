import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter



def find_commons_in_submissions(*file):
    a, b, c, d, e = file
    f1 = pd.read_csv(a)
    f2 = pd.read_csv(b)
    f3 = pd.read_csv(c)
    f4 = pd.read_csv(d)
    f5 = pd.read_csv(e)


    users = f1['user'].unique()

    print('making duplicates...') # 2~3분 걸리는듯?
    df = []
    for user in users:
        a_item = list(f1[f1['user'] == user]['item'].values)
        b_item = list(f2[f2['user'] == user]['item'].values)
        c_item = list(f3[f3['user'] == user]['item'].values)
        d_item = list(f4[f4['user'] == user]['item'].values)
        e_item = list(f5[f5['user'] == user]['item'].values)

        common = a_item + b_item + c_item + d_item + e_item# & c
        top10 = dict(Counter(common).most_common(10)).keys() # Counter 사용해서, 비기는 경우가 있을 때 그냥 이름 순으로 뽑힐듯(여기선)

        for t in top10:
            df.append([user, t])

    new_sub = pd.DataFrame(df, columns=['user', 'item'])

    print('finished!')

    print(new_sub.head(15))

    return users, new_sub



# cwd = os.getcwd()
# files = os.listdir(cwd)
# print("Files in %r: %s" % (cwd, files))


# 비교하고 싶은 csv 파일 주소를 넣자 -> 더 많은 파일을 동시에 비교할 수 있다.
# 파일에서 run 하면 relative-path 를 참조하지 못해 파일을 못읽을 수 있기 때문에, 터미널에서 실행 권장

file1 = '../data/train/16.csv'
file2 = '../data/train/1577.csv'
file3 = '../data/train/1488.csv'
file4 = '../data/train/1450.csv'
file5 = '../data/train/1384.csv'
a = pd.read_csv(file2)
print(a.head())

users ,new_submission_pd = find_commons_in_submissions(file1, file2, file3, file4, file5) # (유저id, 겹치는 답들(set), 겹치는 답들의 개수)


# submission 내보내기
new_submission_pd.to_csv("../data/eval/sub-test.csv", index=False) # (user : int, top10 : dict_keys)




# print('showing barplot!')

# 유저별 겹치는 데이터 수를 barplot 으로 띄워주기 2분 걸리는듯?
# sns.barplot(
#     data=duplicate,
#     x="user",
#     y="common_length",
#     palette="Blues_d"
# )
# plt.xticks([])
# plt.show()
