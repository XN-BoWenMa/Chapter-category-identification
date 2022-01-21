import torch
import torch.nn as nn
from torch.autograd import Variable
from sent_att_model_me import SentAttNet
from word_att_model_me import WordAttNet
from total_model_me import TotalNet

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length, dropout):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size, dropout)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self.total_net = TotalNet(sent_hidden_size, word_hidden_size, num_classes, dropout)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        # 初始化隐层参数
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state2 = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state_before = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state_before2 = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state_after = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state_after2 = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state_before = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state_before2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state_after = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state_after2 = torch.zeros(2, batch_size, self.sent_hidden_size)

        self.total_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.total_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        # self.word_hidden_state = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.word_hidden_size))
        # self.word_hidden_state2 = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.word_hidden_size))
        # self.sent_hidden_state = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.sent_hidden_size))
        # self.sent_hidden_state2 = nn.init.xavier_uniform_(torch.empty(2, batch_size, self.sent_hidden_size))
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.word_hidden_state2 = self.word_hidden_state2.cuda()
            self.word_hidden_state_before = self.word_hidden_state_before.cuda()
            self.word_hidden_state_before2 = self.word_hidden_state_before2.cuda()
            self.word_hidden_state_after = self.word_hidden_state_after.cuda()
            self.word_hidden_state_after2 = self.word_hidden_state_after2.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()
            self.sent_hidden_state2 = self.sent_hidden_state2.cuda()
            self.sent_hidden_state_before = self.sent_hidden_state_before.cuda()
            self.sent_hidden_state_before2 = self.sent_hidden_state_before2.cuda()
            self.sent_hidden_state_after = self.sent_hidden_state_after.cuda()
            self.sent_hidden_state_after2 = self.sent_hidden_state_after2.cuda()
            self.total_hidden_state = self.total_hidden_state.cuda()
            self.total_hidden_state2 = self.total_hidden_state2.cuda()

    def forward(self, input, before, after):
        output_list = []
        final_list = []
        input = input.permute(1, 0, 2) #将batch_size放在第二维度
        for each_sen in input: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state,self.word_hidden_state2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state,self.word_hidden_state2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state,self.sent_hidden_state2))
        final_list.append(output.unsqueeze(0))

        before = before.permute(1, 0, 2)
        for each_sen in before: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_before,self.word_hidden_state_before2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_before,self.word_hidden_state_before2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_before, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_before,self.sent_hidden_state_before2))
        final_list.append(output_before.unsqueeze(0))

        after = after.permute(1, 0, 2)
        for each_sen in after: #each_sen: [batch_size, seq_length]
            output, (self.word_hidden_state_after,self.word_hidden_state_after2) = self.word_att_net(each_sen.permute(1, 0),(self.word_hidden_state_after,self.word_hidden_state_after2))
            output_list.append(output) # output_list：1, batch size, 2*hid_dim
        output = torch.cat(output_list, 0) #体现层次，从单词到句子 max_sent_len, batch size, 2*hid_dim （生成矩阵）
        output_after, (h_output, c_output) = self.sent_att_net(output, (self.sent_hidden_state_after,self.sent_hidden_state_after2))
        final_list.append(output_after.unsqueeze(0))
        
        final_input = torch.cat(final_list, 0)
        # print(final_input.size())
        output, (h_output, c_output) = self.total_net(final_input, (self.total_hidden_state,self.total_hidden_state2))
        return output