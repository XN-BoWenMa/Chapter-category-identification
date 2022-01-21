import torch
import sys
import csv
# csv.field_size_limit(sys.maxsize) #将读取范围调至最大
from sklearn.metrics import classification_report
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
import numpy as np

def get_evaluation(y_true, y_prob, list_metrics):
    # data_path = "D:\\训练数据\\network_test_around_title_2.csv"
    # with open(data_path, 'r', encoding="utf-8-sig") as csv_file:
    #     reader = csv.reader(csv_file) #将csv数据按行读下来
    #     rows = list(reader)
    #     csv_file.close()   
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    if 'report' in list_metrics:
        y_true = y_true.tolist()
        y_pred = y_pred.tolist()
        label_dict = {0:"introduction",1:"method",2:"evaluation and result",3:"related work",4:"conclusion"}
        true_labels = [label_dict[i] for i in y_true]
        predicted_labels = [label_dict[i] for i in y_pred]
        # anti_label_dict = {1:"method",2:"evaluation and result"}
        # filename = r'C:\Users\诸葛绝才\Desktop\appendix.csv'
        # with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:   
        #     csvwriter = csv.writer(write_file, dialect='excel')
        #     for i in range(len(predicted_labels)):
        #         if predicted_labels[i]=="method" or predicted_labels[i]=="evaluation and result":
        #             info = rows[i]
        #             csvwriter.writerow(info)
        output['report'] = classification_report(true_labels, predicted_labels, digits=4)

    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])#增加偏执项
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path): #确定应该使用的最大长度
    word_length_list = []
    sent_length_list = []
    with open(data_path, 'r', encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        for idx, line in enumerate(reader):
            sent_list = line[3].split("\t")
            sent_length_list.append(len(sent_list)) #将每则语料的句子数量进行输入
        for sent in sent_list:
            word_list = sent.split(" ") #将每则语料中的句子进行遍历，将每个句子的单词数进行记录
            word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list) #按照顺序进行sort
        sorted_sent_length = sorted(sent_length_list)
        #取占到80%样本的数量当做最终的长度
    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]

if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print (word)
    print (sent)
