import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
from collections import Counter



def find_commons_in_submissions(file):
    a=file
    year_df=  pd.read_csv('years.tsv',sep='\t')
    df_user_year= pd.read_csv('user_year.csv')
    
    f1 = pd.read_csv(a)
    
    dict_year=year_df.set_index('item').T.to_dict('list')
    dict_user_year=df_user_year.set_index('user').T.to_dict('list')
    users = f1['user'].unique()
    cnt=0
    cnt2=0
    print('Start') # 2~3분 걸리는듯?
    df = []
    for user in users:
        a_item = list(f1[f1['user'] == user]['item'].values)

        new_a_item=[]    
        for i in range(len(a_item)):
            cnt2=cnt2+1
            if int(dict_year.get(a_item[i])[0])>int(dict_user_year.get(user)[2]):
                cnt=cnt+1
    print(file)
    print("movie after last evalution time:",cnt)
    print("ratio:",cnt/cnt2)
    print("end")

    return cnt



# cwd = os.getcwd()
# files = os.listdir(cwd)
# print("Files in %r: %s" % (cwd, files))


# 비교하고 싶은 csv 파일 주소를 넣자 -> 더 많은 파일을 동시에 비교할 수 있다.
# 파일에서 run 하면 relative-path 를 참조하지 못해 파일을 못읽을 수 있기 때문에, 터미널에서 실행 권장

file1 = 'csv/EASE_500-0.1600.csv'
a = pd.read_csv(file1)
# print(a.head())

new_submission_pd = find_commons_in_submissions(file1) # (유저id, 겹치는 답들(set), 겹치는 답들의 개수)

