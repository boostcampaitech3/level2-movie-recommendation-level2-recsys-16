import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from modules import Encoder, LayerNorm

import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import ndcg_k, recall_at_k, neg_sample

class BertDataset(Dataset):
    def __init__(self, args, user_seq, data_type="train"):
        self.user_train = user_seq
        self.num_user = len(user_seq)
        self.num_item = args.item_size -2
        self.max_len = args.max_seq_length
        self.mask_prob = args.mask_p
        self.data_type = data_type
        self.args = args
        
    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        # iterator를 구동할 때 사용됩니다.
        assert self.data_type in {"train", "finetune", "valid", "test", "submission"}
        seq = self.user_train[user]
        neg_items = []
        item_set = set(seq)
        if self.data_type=="train":
            tokens = []
            labels = []
            for s in seq[:-1]:
                prob = np.random.random() # TODO1: numpy를 사용해서 0~1 사이의 임의의 값을 샘플링하세요.
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    # BERT 학습
                    if prob < 0.8:
                        # masking
                        tokens.append(self.args.mask_id)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                    elif prob < 0.9:
                        tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                    else:
                        tokens.append(s)
                    labels.append(s)  # 학습에 사용
                    neg_items.append(neg_sample(item_set, self.args.item_size))
                else:
                    tokens.append(s)
                    labels.append(0)  # 학습에 사용 X, trivial
                    neg_items.append(s)
            tokens.append(self.args.mask_id)
            labels.append(seq[-1])
            neg_items.append(neg_sample(item_set, self.args.item_size))
        
        else:
            if self.data_type == "finetune":
                tmp = seq[:-3]
                tokens = tmp + [self.args.mask_id]
                labels = tmp + [seq[-3]]
                
            elif self.data_type == "valid":
                tmp = seq[:-2]
                tokens = tmp + [self.args.mask_id]
                labels = tmp + [seq[-2]]

            elif self.data_type == "test":
                tmp = seq[:-1]
                tokens = tmp + [self.args.mask_id]
                labels = tmp + [seq[-1]]
            else:
                tokens = seq[:]
                labels = []
            
            neg_items = []
            seq_set = set(tokens)
            
            for _ in tokens:
                neg_items.append(neg_sample(seq_set, self.args.item_size))
                
        pad_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels
        neg_items = [0] * pad_len + neg_items
        
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_items = neg_items[-self.max_len :]
        
        return user, torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(neg_items)


class SeqDataset2(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len, mask_prob, repeat=2):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.repeat = repeat
        self.x_ = []
        self.y_ = []
        for u in self.user_train:
            seq = self.user_train[u]
            for _ in range(self.repeat):
                tokens = []
                labels = []
                for s in seq:
                    prob = np.random.random() # TODO1: numpy를 사용해서 0~1 사이의 임의의 값을 샘플링하세요.
                    if prob < self.mask_prob:
                        prob /= self.mask_prob

                        # BERT 학습
                        if prob < 0.8:
                            # masking
                            tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                        elif prob < 0.9:
                            tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                        else:
                            tokens.append(s)
                        labels.append(s)  # 학습에 사용
                    else:
                        tokens.append(s)
                        labels.append(0)  # 학습에 사용 X, trivial
                tokens = tokens[-self.max_len:]
                labels = labels[-self.max_len:]
                mask_len = self.max_len - len(tokens)

                # zero padding
                tokens = [0] * mask_len + tokens
                labels = [0] * mask_len + labels
                
                self.x_.append(torch.LongTensor(tokens))
                self.y_.append(torch.LongTensor(labels))
    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user*self.repeat

    def __getitem__(self, i): 
        # iterator를 구동할 때 사용됩니다.
        
        return self.x_[i], self.y_[i]
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate) # dropout rate

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate) # scaled dot product attention module을 사용하여 attention 계산
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)
        
        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) 
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() 
        output = output.view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        
        # SASRec과의 dimension 차이가 있습니다.
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units) 
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x)))) # activation: relu -> gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output
    
class BERT4RecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist
    
class BERTModel(nn.Module):
    def __init__(self, args):
        super(BERTModel, self).__init__()
        self.args = args

        self.num_item = args.item_size - 2
        self.hidden_units = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_layers = args.num_hidden_layers
        self.item_encoder = Encoder(args)
        
        self.item_emb = nn.Embedding(self.num_item+2, self.hidden_units, padding_idx = 0) # TODO2: mask와 padding을 고려하여 embedding을 생성해보세요.
        self.pos_emb = nn.Embedding(args.max_seq_length, self.hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.emb_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-6)
        
        self.blocks = nn.ModuleList([BERT4RecBlock(self.num_heads, self.hidden_units, args.attention_probs_dropout_prob) for _ in range(self.num_layers)])
        # TODO3: 예측을 위한 output layer를 구현해보세요. (num_item 주의)
        self.out = nn.Linear(self.hidden_units, self.num_item + 1, bias = True)
        
        self.apply(self._init_weights)
        

    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.pos_emb(position_ids)
        item_emb = self.item_emb(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.emb_layernorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        item_encoded_layers = self.item_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        # mask = torch.tensor(item_seq > 0, device=item_seq.device).unsqueeze(1).repeat(1, item_seq.shape[1], 1).unsqueeze(1)
        # for block in self.blocks:
        #     input_emb, attn_dist = block(input_emb, mask)
        # predictions = (batch_size * max_len * hidden_units)
        predictions = item_encoded_layers[-1]

        # rating_pred = self.predict_full(predictions)
        # rating_pred = self.out(predictions)
        
        return predictions
    
    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.item_emb.weight[:self.num_item+1]
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
class BertTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        finetune_dataset,
        eval_dataloader,
        test_dataloader,
        submission_dataloader,
        args,
    ):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.finetune_dataset = finetune_dataset
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.submission_dataloader = submission_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        

    def train(self, epoch):
        # if self.args.sweep==False:
        #     wandb.watch(self.model, self.model.criterion, log="parameters", log_freq=self.args.log_freq)
        dataloader = self.train_dataloader if self.args.mode=="train" else self.finetune_dataset
        train_data_iter = tqdm(
            enumerate(dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        self.model.train()
        
        if self.args.mode=="train" or "finetune":
            for i, batch in train_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, log_seqs, labels, neg = batch
                # Binary cross_entropy
                logits = self.model(log_seqs)
                # size matching

                self.optim.zero_grad()
                loss = self.cross_entropy(logits, labels, neg)
                loss.backward()
                self.optim.step()

                train_data_iter.set_description(f'Epoch: {epoch:3d}| Train loss: {loss:.5f}')

                post_fix = {
                    "Train loss": round(loss.item(),4),
                }
                # wandb.log(post_fix, step=epoch)
#         else:
#             pred_list = None
#             answer_list = None
#             for i, batch in train_data_iter:
#                 # 0. batch_data will be sent into the device(GPU or CPU)
#                 batch = tuple(t.to(self.device) for t in batch)
#                 user_ids, log_seqs, labels, neg = batch
#                 predictions = self.model(log_seqs)

#                 predictions = predictions[:, -1, :] #[batch_size*(item_num+1)]

#                 predictions = self.predict_full(recommend_output)


#                 predictions = predictions.cpu().data.numpy().copy()
#                 batch_user_index = user_ids.cpu().numpy()
#                 predictions[self.args.train_matrix[batch_user_index].toarray()[:,:-1] > 0] = 0

#                 ind = np.argpartition(predictions, -10)[:, -10:]

#                 arr_ind = predictions[np.arange(len(predictions))[:, None], ind]

#                 arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predictions)), ::-1]

#                 batch_pred_list = ind[
#                     np.arange(len(predictions))[:, None], arr_ind_argsort
#                 ]
#                 if i == 0:
#                     pred_list = batch_pred_list
#                     answer_list = labels.cpu().data.numpy()
#                 else:
#                     pred_list = np.append(pred_list, batch_pred_list, axis=0)
#                     answer_list = np.append(
#                         answer_list, labels.cpu().data.numpy(), axis=0
#                     )
#             score, post_fix = self.get_full_sort_score(epoch, answer_list, pred_list)
            # wandb.log(post_fix, step=epoch)

        if (epoch + 1) % self.args.log_freq == 0:
            print(str(post_fix))

    def valid(self, epoch, mode="valid"):
        self.model.eval()
        pred_list = None
        answer_list = None
        eval_data_iter = tqdm(
            enumerate(self.eval_dataloader),
            desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
            total=len(self.eval_dataloader),
            bar_format="{l_bar}{r_bar}",
        )

        NDCG = 0.0 # NDCG@10
        HIT = 0.0 # HIT@10

        ans = []
        for i, batch in eval_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, log_seqs, labels, neg = batch
        
            predictions = self.model(log_seqs)

            predictions = predictions[:, -1, :]

            predictions = self.predict_full(predictions)
           

            predictions = predictions.cpu().data.numpy().copy()
            batch_user_index = user_ids.cpu().numpy()
            predictions[self.args.train_matrix[batch_user_index].toarray()[:,:-1] > 0] = 0

            ind = np.argpartition(predictions, -10)[:, -10:]

            arr_ind = predictions[np.arange(len(predictions))[:, None], ind]

            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predictions)), ::-1]

            batch_pred_list = ind[
                np.arange(len(predictions))[:, None], arr_ind_argsort
            ]
            
            if i == 0:
                pred_list = batch_pred_list
                answer_list = labels.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(
                    answer_list, labels.cpu().data.numpy(), axis=0
                )

        if mode == "submission":
            print(pred_list)
            return pred_list
        else:
            score, metrics = self.get_full_sort_score(epoch, answer_list, pred_list)
            # wandb.log(metrics, step=epoch)
            return score, metrics

    def test(self, epoch, mode="test"):
        return self.valid(epoch)

    def submission(self, epoch, mode="submission"):
        return self.valid(epoch)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            # "Epoch": epoch,
            "RECALL@5": round(recall[0],4),
            # "NDCG@5": round(ndcg[0],4),
            "RECALL@10": round(recall[1],4),
            "NDCG@10": round(ndcg[1],4),
        }
        print(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], post_fix

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        # self.model = torch.load(file_name)
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_emb(pos_ids)
        neg_emb = self.model.item_emb(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss
    
    def predict_full(self, seq_out):
        # seq_out = self.gather_indexes(seq_out, self.args.max_seq_length)
        # [item_num hidden_size]
        test_item_emb = self.model.item_emb.weight[:self.model.num_item+1]
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)