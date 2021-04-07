# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import copy
import math
import numpy as np


class CNN_Encoder(nn.Module):
    """This is the cnn model---resnet101"""
    def __init__(self, encoder_weights=None):
        super(CNN_Encoder, self).__init__()
        resnet = torchvision.models.resnet101(True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pooling = nn.MaxPool2d(14, 14)
        self.fc = nn.Linear(2048, 80)

    def forward(self, x):
        return_tuple = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return_tuple.append(x.permute(0, 2, 3, 1))

        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return_tuple.append(x)
        return return_tuple


    
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, src_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.fc1 = nn.Linear(512*14*14, 80)
        self.fc_2 = nn.Linear(512*14*14, 300)
        
    def forward(self, fc_feats, src, src_mask,):
        """Take in and process masked src and target sequences."""
        batch_size = src.size(0)
        encoder_out = self.encoder(src, src_mask) 
        
        ## for embedding
        x = self.fc_2(encoder_out.view(batch_size, -1))
        ## for classification
        output = self.fc1(encoder_out.view(batch_size, -1))
        return output.squeeze(), x
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x_new = self.self_attn(x, x, x, mask)
        x = x + self.dropout(self.norm(x_new))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerModel(nn.Module):
    """Transformer encoder model main function"""
    def make_model(self, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            lambda x:x)
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__()
        self.opt = opt
        self.N_enc = opt.num_layers
        self.N_dec = opt.num_layers
        self.d_model = opt.embed_size
        self.h = 8
        self.dropout = opt.dropout
        self.input_fea_size = opt.input_fea_size
        
        self.embed = nn.Sequential(nn.Embedding(opt.vocab_size, self.d_model),
                                nn.ReLU(),
                                nn.Dropout(self.dropout))
        
        self.fc_embed = nn.Sequential(nn.Linear(self.input_fea_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout))

        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.input_fea_size),) if self.opt.use_bn else ())+
                                    (nn.Linear(self.input_fea_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.opt.use_bn==2 else ())))

        self.emb_vocab = nn.Sequential()

        self.model = self.make_model(N_enc=self.N_enc, N_dec=self.N_dec, d_model=self.d_model, d_ff=self.opt.d_ff,  h=self.h, dropout=self.dropout)


    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats = self.att_embed(att_feats)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        return att_feats, att_masks

    def forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, att_masks = self._prepare_feature_forward(att_feats, att_masks)
        fc_feats = self.fc_embed(fc_feats)

        cls_out, x = self.model(fc_feats, att_feats, att_masks)

        return cls_out, x
