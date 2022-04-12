import argparse
import torch
import numpy as np
import pandas as pd

from logging import getLogger
from tqdm import tqdm
from recbole.utils import init_logger, get_model, init_seed, set_color 
from recbole.data import Interaction, data_preparation
from recbole.data.utils import create_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/model.pth', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-rec', help='name of dataset')
    parser.add_argument('--rank_num', '-k', type=int, default=10, help='num of ranking K')
    
    args, _ = parser.parse_known_args()
    
    # rank K 설정
    K = args.rank_num 

    # config, model, dataset 불러오기
    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']
    config['dataset'] = args.dataset

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, test_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    # device 설정
    device = config.final_config_dict['device']
    
    # user, item id -> token 변환 array
    user_id = config['USER_ID_FIELD']
    item_id = config['ITEM_ID_FIELD']
    user_id2token = dataset.field2id_token[user_id]
    item_id2token = dataset.field2id_token[item_id]

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(-1,128)
    
    # user, item 길이
    user_len = len(user_id2token)
    item_len = len(item_id2token)

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None
    
    # model 평가모드 전환
    model.eval()
    
    # progress bar 설정
    tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink'))

    for data in tbar:
        # interaction 생성
        interaction = dict()
        interaction = Interaction(interaction)
        interaction[user_id] = data
        interaction = interaction.to(device)

        # user item별 score 예측
        score = model.full_sort_predict(interaction)
        score = score.view(-1, item_len)
        
        rating_pred = score.cpu().data.numpy().copy()
        
        user_index = data.numpy()
        
        idx = matrix[user_index].toarray() > 0

        rating_pred[idx] = -np.inf
        rating_pred[:, 0] = -np.inf
        ind = np.argpartition(rating_pred, -K)[:, -K:]
        
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[
            np.arange(len(rating_pred))[:, None], arr_ind_argsort
        ]
        
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(
                user_list, user_index, axis=0
            )
    
    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 데이터 저장
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by='user', inplace=True)
    dataframe.to_csv(
        "submission.csv", index=False
    )
    print('inference done!')
