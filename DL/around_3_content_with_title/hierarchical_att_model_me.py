import torch
import torch.nn as nn
from torch.autograd import Variable
from sent_att_model_me import SentAttNet
from word_att_model_me import WordAttNet
from total_model_me import TotalNet
from total_model_me_cnn import TotalNet_CNN
from total_model_me_final import TotalNet_Final
from cnn_me import CnnNet


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length, n_filters, filter_sizes, dropout):
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
        self.total_net_cnn = TotalNet_CNN(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self.total_net_final = TotalNet_Final(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        # 初始化隐层参数
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_before1_1 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_before1_2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_before2_1 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_before2_2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_before3_1 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_before3_2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_after1_1 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_after1_2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_after2_1 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_after2_2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_after3_1 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.word_hidden_state_after3_2 = torch.zeros(2, batch_size, self.word_hidden_size).cuda()
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_before1_1 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_before1_2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_before2_1 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_before2_2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_before3_1 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_before3_2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_after1_1 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_after1_2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_after2_1 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_after2_2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_after3_1 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.sent_hidden_state_after3_2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        # 整合特征后的特征向量
        self.total_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.total_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.total_hidden_state_cnn = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        self.total_hidden_state2_cnn = torch.zeros(2, batch_size, self.sent_hidden_size).cuda()
        # self.word_hidden_state = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.word_hidden_size))
        # self.word_hidden_state2 = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.word_hidden_size))
        # self.sent_hidden_state = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.sent_hidden_size))
        # self.sent_hidden_state2 = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.sent_hidden_size))
        # if torch.cuda.is_available():

    def forward(self, input, before1, before2, before3, after1, after2, after3, title, title_before1, title_before2, title_before3, title_after1, title_after2, title_after3):
        final_list = []

        before3 = before3.permute(1, 0, 2)
        output_list = []
        for each_sen in before3: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_before3_1,self.word_hidden_state_before3_2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_before3_1,self.word_hidden_state_before3_2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_before, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_before3_1,self.sent_hidden_state_before3_2))
        
        before_feature = self.cnn_net(title_before3)
        input_before3 = torch.cat((output_before, before_feature.float()), dim=1)
        final_list.append(input_before3.unsqueeze(0))
        #---------------------------------------------------------
        
        before2 = before2.permute(1, 0, 2)
        output_list = []
        for each_sen in before2: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_before2_1,self.word_hidden_state_before2_2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_before2_1,self.word_hidden_state_before2_2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_before, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_before2_1,self.sent_hidden_state_before2_2))
        
        before_feature = self.cnn_net(title_before2)
        input_before2 = torch.cat((output_before, before_feature.float()), dim=1)
        final_list.append(input_before2.unsqueeze(0))
        #---------------------------------------------------------
        
        before1 = before1.permute(1, 0, 2)
        output_list = []
        for each_sen in before1: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_before1_1,self.word_hidden_state_before1_2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_before1_1,self.word_hidden_state_before1_2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_before, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_before1_1,self.sent_hidden_state_before1_2))
        
        before_feature = self.cnn_net(title_before1)
        input_before1 = torch.cat((output_before, before_feature.float()), dim=1)
        final_list.append(input_before1.unsqueeze(0))
        #---------------------------------------------------------
        input = input.permute(1, 0, 2) #将batch_size放在第二维度
        output_list = []
        for each_sen in input: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state,self.word_hidden_state2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state,self.word_hidden_state2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state,self.sent_hidden_state2))
        
        title_feature = self.cnn_net(title)
        input_current = torch.cat((output, title_feature.float()), dim=1)
        final_list.append(input_current.unsqueeze(0))

        #---------------------------------------------------------
        after1 = after1.permute(1, 0, 2)
        output_list = []
        for each_sen in after1: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_after1_1,self.word_hidden_state_after1_2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_after1_1,self.word_hidden_state_after1_2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_after, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_after1_1,self.sent_hidden_state_after1_2))    

        after_feature = self.cnn_net(title_after1)
        input_after1 = torch.cat((output_after, after_feature.float()), dim=1)
        final_list.append(input_after1.unsqueeze(0))

        #---------------------------------------------------------
        after2 = after2.permute(1, 0, 2)
        output_list = []
        for each_sen in after2: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_after2_1,self.word_hidden_state_after2_2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_after2_1,self.word_hidden_state_after2_2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_after, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_after2_1,self.sent_hidden_state_after2_2))    

        after_feature = self.cnn_net(title_after2)
        input_after2 = torch.cat((output_after, after_feature.float()), dim=1)
        final_list.append(input_after2.unsqueeze(0))

        #---------------------------------------------------------
        after3 = after3.permute(1, 0, 2)
        output_list = []
        for each_sen in after3: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_after3_1,self.word_hidden_state_after3_2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_after3_1,self.word_hidden_state_after3_2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_after, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_after3_1,self.sent_hidden_state_after3_2))    

        after_feature = self.cnn_net(title_after3)
        input_after3 = torch.cat((output_after, after_feature.float()), dim=1)
        final_list.append(input_after3.unsqueeze(0))
        
        final_input = torch.cat(final_list, 0)
        output, (h_output, c_output) = self.total_net(final_input, (self.total_hidden_state,self.total_hidden_state2))
        
        return output