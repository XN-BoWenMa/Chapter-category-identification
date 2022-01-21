import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_me import matrix_mul, element_wise_mul
from torch.autograd import Variable

class TotalNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14, dropout=0.5, n_layers=1):
        super(TotalNet, self).__init__()

        self.lstm = nn.LSTM(2 * word_hidden_size, sent_hidden_size, num_layers=n_layers, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        # self._create_weights2()

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
    
    def _create_weights2(self):

        self.sent_weight.data.uniform_(-0.1, 0.1)
        self.context_weight.data.uniform_(-0.1,0.1)

    def forward(self, input, hidden_state):
        f_output, (h_output, c_output) = self.lstm(input.float(), hidden_state)
        # f_output = self.dropout(f_output)

        output = torch.cat((h_output[-2,:,:], h_output[-1,:,:]), dim=1)

        # output = self.dropout(output)
        output = self.fc(output)#相当于全连接层

        return output, (h_output, c_output)



if __name__ == "__main__":
    abc = SentAttNet()