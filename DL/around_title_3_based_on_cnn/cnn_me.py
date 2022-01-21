import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
import torch.nn.functional as F

class CnnNet(nn.Module):
    def __init__(self, word2vec_path, n_filters, filter_sizes, dropout):
        super().__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        #将<unk>、<pad>标签默认为0向量
        pad_word = np.zeros((1, embed_size))
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))#将unknown_word拼接到dict前面，索引为1
        dict = torch.from_numpy(np.concatenate([pad_word, dict], axis=0).astype(np.float))#将pad_word拼接到dict前面，索引为0
        self.filter_sizes = filter_sizes #过滤器的尺寸list
        self.embedding = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                              kernel_size = (fs, embed_size)) 
                                    for fs in filter_sizes
                                    ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # CHANGED
        # mask = (text == 1).bool()
        # print(text)
        embedded = self.embedding(text) # [batch size, sent len, emb dim] 不需要进行packed处理
        # print(embedded.size())
        embedded = embedded.unsqueeze(1) # [batch size, 1, sent len, emb dim]
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
        return cat