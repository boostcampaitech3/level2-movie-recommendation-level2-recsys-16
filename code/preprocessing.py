import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default='S3RecModel', help="which model to use training"
    )
    args = parser.parse_args()

    if args.model == 'S3RecModel':
        item_df = pd.read_csv('../data/train/titles.tsv', sep='\t')
        item_ids = item_df['item'].unique()
        item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)

        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        genres_df['item'] = genres_df['item'].map(lambda x: item2idx[x])

        array, index = pd.factorize(genres_df["genre"])
        genres_df["genre"] = array
        genres_df.groupby("item")["genre"].apply(list).to_json(
            "../data/Ml_item2attributes.json"
        ) # json 파일로 내보내기

        item2idx.to_csv('../data/item2idx.tsv', sep='\t', encoding='utf-8', index=True)

    elif args.model == 'FM_pair':
        ratings_df = pd.read_csv('../data/train/train_ratings.csv')
        genres_df = pd.read_csv('../data/train/genres.tsv', sep='\t')
        title_df = pd.read_csv('../data/train/titles.tsv', sep='\t')
        director_df = pd.read_csv('../data/train/directors.tsv', sep='\t')
        writer_df = pd.read_csv('../data/train/writers.tsv', sep='\t')
        year_df = pd.read_csv('../data/train/years.tsv', sep='\t')
        year_df = year_df.set_index(['item'], drop=False)

        def get_year(x):
            if x not in year_df.index:
                return np.nan
            else:
                return int(year_df.loc[x]['year'])

        # 중복된 영화 평가 제거 하기(# 가장 일찍 평가된 애들을 남기기) -> eda 결과 한사람이 하나의 영화에 여러번 평가 x
        ratings_df = ratings_df.drop_duplicates(['user', 'item'])

        # 아이템 데이터셋 합쳐놓기
        total_df = title_df.copy()
        total_df['genres'] = total_df['item'].apply(lambda x: list(genres_df[genres_df['item'] == x]['genre'].values))
        total_df['directors'] = total_df['item'].apply(
            lambda x: list(director_df[director_df['item'] == x]['director'].values))
        total_df['writers'] = total_df['item'].apply(lambda x: list(writer_df[writer_df['item'] == x]['writer'].values))
        total_df['year'] = total_df['item'].apply(lambda x: get_year(x))

        # csv 파일로 내보내기
        total_df.to_csv("../data/train/full_item.tsv")




if __name__ == "__main__":
    main()


