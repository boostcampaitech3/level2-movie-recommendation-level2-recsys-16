import argparse
import os
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    get_FM_data,
)
from rankfm.rankfm import RankFM
from rankfm.evaluation import hit_rate, reciprocal_rank, discounted_cumulative_gain, precision, recall, diversity
# pip install rankfm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # model args
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--using_pretrain", action="store_true")

    parser.add_argument( # 어떤 모델로 학습을 시작할지 적어놓자
        "--model", type=str, default='S3RecModel', help="which model to use training"
    )

    parser.add_argument(  # 어떤 모델로 학습을 시작할지 적어놓자
        "--k", type=int, default='10', help="k in FM's validation"
    )

    args = parser.parse_args()

    # --------------------
    set_seed(args.seed)
    check_path(args.output_dir)
    print('checked output path')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print(f'Cuda use: {torch.cuda.is_available()}')

    # ----------------------------- 데이터 파일 체크 ( FM 에서 사용할 데이터 'ex 장르..' 추가 )
    args.data_file = args.data_dir + "train_ratings.csv"
    args.data_genres = args.data_dir + 'genres.tsv'
    args.full_items = args.data_dir + 'full_item.tsv'
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    print('set data, attribute files!')

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # save model, Sampler 부분
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)


    # 데이터 파일 로드 - S3RecModel 용
    if args.model == 'S3RecModel':
        user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
            args.data_file
        )

        item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
        args.attribute_size = attribute_size + 1


        args.item2attribute = item2attribute
        # set item score in train set to `0` in validation
        args.train_matrix = valid_rating_matrix


        train_dataset = SASRecDataset(args, user_seq, data_type="train")
        train_sampler = RandomSampler(train_dataset) # 배치를 빼낼때 섞어서 빼낸다 -> user 들을 섞는 것
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )

        eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset) # 배치를 빼낼때 순차적으로 빼낸다
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
        )

        test_dataset = SASRecDataset(args, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset) # 배치를 빼낼때 순차적으로 빼낸다
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.batch_size
        )

        #---------------------- 모델 설정하는 부분---------#
        model = S3RecModel(args=args)
        print(type(model))
        trainer = FinetuneTrainer(   # 트레이너 안에, 모델 하나와, 데이터 로더들, None, 그리고 args 를 넣는다
            model, train_dataloader, eval_dataloader, test_dataloader, None, args
        )

        print(args.using_pretrain)
        ### pretrain 을 사용한다면 이 부분을 이용한다
        if args.using_pretrain:
            pretrained_path = os.path.join(args.output_dir, "Pretrain.pt") # pretrain.pt 지정한거로 바꿔서 돌리자
            try:
                trainer.load(pretrained_path)
                print(f"Load Checkpoint From {pretrained_path}!")

            except FileNotFoundError:
                print(f"{pretrained_path} Not Found! The Model is same as SASRec")
        else:
            print("Not using pretrained model. The Model is same as SASRec")

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)

            scores, _ = trainer.valid(epoch)

            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)
        print(result_info)

    #------------------------------------------------------ RankFM

    elif args.model == 'FM':



        interaction, item_feature, user_feature = get_FM_data(args.data_file, args.data_genres, args.full_items) # FM 학습에서 사용할 데이터 세팅

        # 모델 설정 --> args 로 받아서 설정할 수 있는 부분은 설정하도록
        # factors = latent factor rank, [user, item, user-feature, item-feature]

        model = RankFM(factors=15, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=args.lr,
                       learning_schedule='invscaling')  # 상속받은 모델, invscaling: 학습률을 점점 줄여나감
        # # wandb 연결
        # wandb.login()
        # with wandb.init(project="RankFM", entity="recsys16", config=vars(args)):
        #     args = wandb.config
        #     model = model
        #     # early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        #
        # # 훈련시키기
        #     for epoch in range(args.epochs):
        #         print('current epoch:', epoch)
        #         # 데이터 셋(user_id/item_id)을 train/valid 로 나눈다
        #         interaction['random'] = np.random.random(size=len(interaction))
        #         test_pct = 0.20
        #         train_mask = interaction['random'] < (1 - test_pct)
        #         valid_mask = interaction['random'] >= (1 - test_pct)
        #
        #         interactions_train = interaction[train_mask][['user_id', 'item_id']]
        #         interactions_valid = interaction[valid_mask][['user_id', 'item_id']]
        #
        #         # 존재하는 아이템 벡터만 사용하기 위해서
        #         train_items = np.sort(interactions_train.item_id.unique())
        #         valid_items = np.sort(interactions_valid.item_id.unique())
        #         cold_start_items = set(valid_items) - set(train_items)
        #         print('cold start items: ', cold_start_items)
        #
        #         train_item_features = item_feature[item_feature.item_id.isin(train_items)]
        #
        #
        #         print('model fitting...')
        #         # 모델 fit 부분
        #         # log likelihood represents user preferences for observed items over unobserved items
        #         model.fit_partial(interactions_train, item_features=train_item_features, epochs=1, verbose=True)
        #
        #         print('model validing...')
        #         # valid_set을 이용하여 모델 성능 평가 -> 체크할 지표를 줄이면 훈련 시간 감소!
        #         print('hit_rate') # 3 ~ 5분 소요
        #         # model_hit_rate = hit_rate(model, interactions_valid, k=args.k)
        #
        #         # print('reiprocal_rank') # 3 ~ 5분 소요
        #         # model_reciprocal_rank = reciprocal_rank(model, interactions_valid, k=args.k)
        #         # print('dcg') # 3 ~ 5분 소요
        #         # model_dcg = discounted_cumulative_gain(model, interactions_valid, k=args.k)
        #         # print('precision') # 3 ~ 5분 소요
        #         # model_precision = precision(model, interactions_valid, k=args.k)
        #         print('recall') # 3 ~ 5분 소요
        #         # model_recall = recall(model, interactions_valid, k=args.k)
        #
        #         # print("hit_rate: {:.3f}".format(model_hit_rate))
        #         # print("reciprocal_rank: {:.3f}".format(model_reciprocal_rank))
        #         # print("dcg: {:.3f}".format(model_dcg, 3))
        #         # print("precision: {:.3f}".format(model_precision))
        #         # print("recall: {:.3f}".format(model_recall))
        #
        #         # wandb.log({'epoch': epoch, 'recall': model_recall, 'hit_rate': hit_rate}) # log 에 담을 값을 자유롭게 추가
        #         # #generate user-item scores from the validation data
        #         # valid_scores = model.predict(interactions_valid, cold_start='nan')
        #
        # ## early-stopping 을 걸자 (loglikelyhood 값이 몇 회 이상 올라가지 않으면 바로 학습 종료하고 recommend code로 넘거가게 끔)
        model.fit_partial(interaction[['user_id', 'item_id']], epochs=1, verbose=True)

        model.predict

        print('model recommending...')


        user_id = user_feature['user_id'].unique()
        # 모든 유저에 대해 top 10 개를 추천한다 --> 결과를 submission 형태로 바꾸어주는 코드도 필요할 듯 하다, inference 시에 submission
        valid_recs = model.recommend(user_id, n_items=10, filter_previous=True, cold_start='drop')
        print('recommend 끝')
        valid_recs.insert(0, 'user_id', valid_recs.index)
        print('insert 끝')
        # valid_recs.to_csv(f"check_{args.model_name}+.csv")
        print('valid_rec 파일 완성') # 잘 나옴 체크 완료
        print(valid_recs.head())



        user_ids = valid_recs['user_id'].unique()
        print('user들의 수',len(user_ids))
        print(user_ids)


        # submission 형태로 csv 파일 만들기
        submission = pd.DataFrame(columns=('user', 'item'))
        for user in user_ids:
            item_ids = list(valid_recs[valid_recs['user_id'] == user].values[0][1:])
            ui = []
            for item in item_ids:
                ui.append([user, item])

            # print(ui)
            df = pd.DataFrame(ui, columns=['user', 'item'])
            submission = pd.concat([submission, df])

        submission.to_csv(f"../data/train/{args.model_name}_submission.csv", index=False) # 제출용 submission file 생성
        print('submission 파일 생성 완료')

if __name__ == "__main__":
    main()

