import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
import math

class MF_basemodel(nn.Module):
    """
    Base module for matrix factorization.
    """
    def __init__(self, n_user, n_item, dim=40, dropout=0, init = None):
        super().__init__()
        
        self.user_latent = nn.Embedding(n_user, dim)
        self.item_latent = nn.Embedding(n_item, dim)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.sigmoid = nn.Sigmoid()
        if init is not None:
            self.init_embedding(init)
        else: 
            self.init_embedding(0)
        
    def init_embedding(self, init): 
        
        nn.init.kaiming_normal_(self.user_latent.weight, mode='fan_out', a = init)
        nn.init.kaiming_normal_(self.item_latent.weight, mode='fan_out', a = init)
          
    def forward(self, users, items):

        u_latent = self.dropout(self.user_latent(users))
        i_latent = self.dropout(self.item_latent(items))

        preds = torch.sum(u_latent * i_latent, dim=1, keepdim=True)
        preds = self.sigmoid(preds)

        return preds.squeeze(dim=-1)

    def l2_norm(self, users, items): 
        users = torch.unique(users)
        items = torch.unique(items)
        
        l2_loss = (torch.sum(self.user_latent(users)**2) + torch.sum(self.item_latent(items)**2)) / 2
        return l2_loss

