import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_me import matrix_mul, element_wise_mul
from torch.autograd import Variable
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50, dropout=0.5, n_layers=1):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        #将<unk>、<pad>标签默认为0向量
        pad_word = np.zeros((1, embed_size))
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))#将unknown_word拼接到dict前面，索引为1
        dict = torch.from_numpy(np.concatenate([pad_word, dict], axis=0).astype(np.float))#将pad_word拼接到dict前面，索引为0
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)
        # self._create_weights2()
        # self.softmax_word = nn.Softmax()
        self.dropout = nn.Dropout(dropout)
        # self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
    
    def _create_weights2(self):

        self.word_weight.data.uniform_(-0.1, 0.1)
        self.context_weight.data.uniform_(-0.1,0.1)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        output = self.dropout(output)
        f_output, (h_output, c_output) = self.lstm(output.float(), hidden_state)  # feature output and hidden state output
        
        output = torch.cat((h_output[-2,:,:], h_output[-1,:,:]), dim=1).unsqueeze(0)

        # output = matrix_mul(f_output, self.word_weight, self.word_bias)#先进行线性输出
        # output = matrix_mul(output, self.context_weight).permute(1,0)#计算相似度
        # output = F.softmax(output,dim=1)#按列softmax
        # output = element_wise_mul(f_output,output.permute(1,0))
        output = self.dropout(output)
        
        return output, (h_output, c_output)


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")