import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_duplicates_in_submissions(file1, file2):
    a = pd.read_csv(file1)
    b = pd.read_csv(file2)

    # 만약 assertion 에러가 발생하면, csv 파일을 읽어오는 중 unnamed: 0 컬럼이 추가된 데이터에
    # index_col=[0] 을 read_csv 할 때 던져준다
    assert a.shape == b.shape, f"shape doesn't match!\na: {a.shape} b: {b.shape}"

    users = a['user'].unique()

    print('making duplicates...') # 2~3분 걸리는듯?
    df = []
    for user in users:
        a_item = set(a[a['user'] == user]['item'].values)
        b_item = set(b[b['user'] == user]['item'].values)
        common = a_item & b_item
        df.append([user, common, len(common)])

    duplicate = pd.DataFrame(df, columns=['user', 'common', 'common_length'])
    print('finished!')
    return duplicate

# 비교하고 싶은 csv 파일 주소를 넣자
file1 = '../data/train/genre_submission.csv'
file2 = '../data/train/genre_writer2_submission.csv'

duplicate = check_duplicates_in_submissions(file1, file2) # (유저id, 겹치는 답들(set), 겹치는 답들의 개수)

print(duplicate.head())
print('showing barplot!')

# 유저별 겹치는 데이터 수를 barplot 으로 띄워주기 2분 걸리는듯?
sns.barplot(
    data=duplicate,
    x="user",
    y="common_length",
    palette="Blues_d"
)
plt.xticks([])
plt.show()
