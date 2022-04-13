import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

ratios = []
dataframe_list = []

print('순위별 가중치값 입력(ex: 1 0.9 0.8 ...)')
rank_ratio = list(map(float, sys.stdin.readline().split()))
rank_len = len(rank_ratio)

num = int(input('앙상블할 모델 결과 개수: '))

for i in range(num):
    filepath = input(f'{i+1}번째 파일 경로: ')
    ratio = float(input(f'{i+1}번째 파일 가중치(0~1 사이 실수): '))
    dataframe_list.append(pd.read_csv(filepath))
    ratios.append(ratio)

user_list = dataframe_list[0]['user'].unique()
dataframe_len = len(dataframe_list)

result = []
tbar = tqdm(user_list, desc='Ensemble')
for user in tbar:
    temp = defaultdict(float)
    for df_idx in range(dataframe_len):
        items = dataframe_list[df_idx][dataframe_list[df_idx]['user'] == user]['item'].values
        max_rank = min(len(items), rank_len)
        for rank_idx in range(max_rank):
            temp[items[rank_idx]] += rank_ratio[rank_idx] * ratios[df_idx]

    for key, _ in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:10]:
        result.append((user, key))

submission = pd.DataFrame(result, columns=['user', 'item'])
submission.to_csv(f'ensemble({num})-file.csv', index=False)