import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
import torch.nn.functional as F

class CnnNet_Final(nn.Module):
    def __init__(self, n_filters, filter_sizes, filter_sizes_fuse, dropout):
        super(CnnNet_Final, self).__init__()
        self.filter_sizes = filter_sizes #过滤器的尺寸list
        self.filter_sizes_fuse = filter_sizes_fuse
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                              kernel_size = (fs, n_filters*len(filter_sizes)+200)) 
                                    for fs in filter_sizes_fuse
                                    ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes_fuse), 5)
        
    def forward(self, text):
        # CHANGED
        # mask = (text == 1).bool()
        # print(text)
        # embedded = self.embedding(text) # [batch size, sent len, emb dim] 不需要进行packed处理
        # print(embedded.size())
        embedded = text.unsqueeze(1) # [batch size, 1, sent len, emb dim]
        # print(embedded.size())
        # for conv in self.convs:
        #     s = F.relu(conv(embedded.float()))
        #     print(s.size())
        #     break
        conved = [F.relu(conv(embedded.float())).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
#         print((1.-mask[:, :-3+1]).unsqueeze(1).byte().shape)
        # conved = [conv.masked_fill(mask[:, :-filter_size+1].unsqueeze(1).bool(), -999999) \
        #           for (conv, filter_size) in zip(conved, self.filter_sizes)]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        # cat = self.dropout(torch.cat(pooled, dim=1))
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        output = self.fc(cat)#相当于全连接层
        return output