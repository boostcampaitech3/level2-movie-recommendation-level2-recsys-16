import pandas as pd
from collections import defaultdict
from tqdm import tqdm

ratios = []
dataframe_list = []

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
    for idx in range(dataframe_len):
        items = dataframe_list[idx][dataframe_list[idx]['user'] == user]['item'].values
        for item in items:
            temp[item] += ratios[idx]

    for key, _ in sorted(temp.items(), key=lambda x: x[1], reverse=True)[:10]:
        result.append((user, key))

submission = pd.DataFrame(result, columns=['user', 'item'])

submission.to_csv(f'ensemble-{num}file', index=False)