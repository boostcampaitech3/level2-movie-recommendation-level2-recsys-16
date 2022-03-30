import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.optim import Adam, lr_scheduler

from utils import ndcg_k, recall_at_k
from modules import *


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__(
        args
        )
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.attribute_embeddings = nn.Embedding(
            args.attribute_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)
        
        self.rec_avg_loss = 0.0
        self.rec_cur_loss = 0.0
    
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=betas,
            weight_decay=self.args.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        rec_data_iter = tqdm.tqdm(
            enumerate(dataloader),
            desc="Recommendation EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}",
        )
            
        # 0. batch_data will be sent into the device(GPU or CPU)
        batch = tuple(t.to(self.device) for t in batch)
        _, input_ids, target_pos, target_neg, _ = batch
        # Binary cross_entropy
        sequence_output = self.model.finetune(input_ids)
        loss = self.cross_entropy(sequence_output, target_pos, target_neg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.rec_avg_loss += loss.item()
        self.rec_cur_loss = loss.item()

        post_fix = {
            "epoch": epoch,
            "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_data_iter)),
            "rec_cur_loss": "{:.4f}".format(rec_cur_loss),
        }

        if (epoch + 1) % self.args.log_freq == 0:
            print(str(post_fix))
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        return self.get_full_sort_score(epoch, answer_list, pred_list)
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        
    def predict_step(self, batch, batch_idx):
        
        return pred_list
    
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    