import torch
import torch.nn as nn
from torch.autograd import Variable
from sent_att_model_me import SentAttNet
from word_att_model_me import WordAttNet
from total_model_me import TotalNet
from cnn_me import CnnNet
from cnn_final import CnnNet_Final
import csv
import random

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length, n_filters, filter_sizes, filter_sizes_fuse, dropout):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size, dropout)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self.cnn_net = CnnNet(pretrained_word2vec_path, n_filters, filter_sizes, dropout)
        self.total_net = TotalNet(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self.cnn_final = CnnNet_Final(n_filters, filter_sizes, filter_sizes_fuse, dropout)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        # 初始化隐层参数
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state2 = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size)

        self.total_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.total_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.total_hidden_state_final = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.total_hidden_state_final2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        # self.word_hidden_state = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.word_hidden_size))
        # self.word_hidden_state2 = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.word_hidden_size))
        # self.sent_hidden_state = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.sent_hidden_size))
        # self.sent_hidden_state2 = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.sent_hidden_size))
        # if torch.cuda.is_available():
        #     self.word_hidden_state = self.word_hidden_state.cuda()
        #     self.word_hidden_state2 = self.word_hidden_state2.cuda()
        #     self.sent_hidden_state = self.sent_hidden_state.cuda()
        #     self.sent_hidden_state2 = self.sent_hidden_state2.cuda()
        #     self.total_hidden_state = self.total_hidden_state.cuda()
        #     self.total_hidden_state2 = self.total_hidden_state2.cuda()
        #     self.total_hidden_state_final = self.total_hidden_state_final.cuda()
        #     self.total_hidden_state_final2 = self.total_hidden_state_final2.cuda()

    def forward(self, title, before1, before2, before3, after1, after2, after3, iter, epoch):
        final_list = []

        before_feature3 = self.cnn_net(before3)

        before_feature2 = self.cnn_net(before2)

        before_feature1 = self.cnn_net(before1)
        
        title_feature = self.cnn_net(title)
        
        after_feature1 = self.cnn_net(after1)

        after_feature2 = self.cnn_net(after2)

        after_feature3 = self.cnn_net(after3)

        sequence = [before_feature3, before_feature2, before_feature1, title_feature, after_feature1, after_feature2, after_feature3]
        for each_seq in sequence:
            final_list.append(each_seq.unsqueeze(1))
        final_input = torch.cat(final_list, 1)

        output = self.cnn_final(final_input)

        return output