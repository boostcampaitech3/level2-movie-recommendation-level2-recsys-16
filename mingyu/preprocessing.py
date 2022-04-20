import pandas as pd
import numpy as np

from utils import item2idx_

def main():
    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    genres_df["item"] = genres_df['item'].map(lambda x: item2idx_[x])
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "../data/Ml_item2attributes.json"
    )
    
    item_df = pd.read_csv('../data/train/titles.tsv', sep='\t')
    item_ids = item_df['item'].unique()
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids)
    item2idx.to_csv('../data/item2idx.tsv', sep='\t', encoding='utf-8', index=True)

if __name__ == "__main__":
    main()
