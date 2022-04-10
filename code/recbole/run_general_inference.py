import argparse
import torch
import numpy as np
import pandas as pd

from logging import getLogger
from tqdm import tqdm
from recbole.sampler import Sampler
from recbole.utils import init_logger, get_model, init_seed, set_color 
from recbole.data import Interaction, data_preparation
from recbole.data.utils import create_dataset, get_dataloader
from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/general_model.pth', help='name of models')
    
    args, _ = parser.parse_known_args()
    
    # config, model, dataset 불러오기
    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    

    config['eval_args']['order'] = 'TO'
    config['eval_args']['split'] = {'RS': [0, 0, 1.0]}
    config['eval_neg_sample_args'] = {'strategy': 'none'}

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
    all_user_list = torch.arange(1, len(user_id2token))
    
    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # user id, predict item id 저장 변수
    pred_list = []
    user_list = []
    
    # model 평가모드 전환
    model.eval()
    
    # progress bar 설정
    tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink'))

    for data in tbar:
        # interaction 생성
        interaction = dict()
        interaction = Interaction(interaction)
        interaction[user_id] = data.unsqueeze(0)
        interaction = interaction.to(device)

        # user item별 score 예측
        score = model.full_sort_predict(interaction)
        if score.ndim == 2: score = score[0]
        
        rating_pred = score.cpu().data.numpy().copy()
        
        user_index = data.numpy()
        
        idx = matrix[user_index].toarray() > 0

        rating_pred[idx[0]] = -np.inf
        rating_pred[0] = -np.inf
        ind = np.argpartition(rating_pred, -10)[-10:]
        
        arr_ind = rating_pred[ind]

        arr_ind_argsort = np.argsort(arr_ind)[::-1]

        batch_pred_list = ind[arr_ind_argsort]
        
        pred_list.append(batch_pred_list)
        user_list.append(user_index)
        
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
