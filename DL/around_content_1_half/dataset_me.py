import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels, around_before = [], [], []
        label_sign = {"introduction":0,"method":1,"evaluation and result":2,"related work":3,"conclusion":4}
        with open(data_path, 'r', encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file) #将csv数据按行读下来
            rows = list(reader)
            csv_file.close()
        for row in rows:
            text = row[3].split("\t")
            label = int(label_sign[row[1]])
            if len(row[4])>0:
                around_before.append(row[4].split("\t"))
            else:
                around_before.append("")
            texts.append(text)
            labels.append(label)

        self.texts = texts
        self.labels = labels
        self.around_before = around_before
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values #usecols=[0]选取第一列数据，即是单词
        self.dict = [word[0] for word in self.dict] #将每个单词放到list中（单词本身）
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))
        # print(self.num_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index): #获取每一条数据 #并进行编码
        label = self.labels[index]
        text = self.texts[index]
        before = self.around_before[index]

        before_encode = []
        if len(before)>0:
            before_encode = [ 
                [self.dict.index(word) if word in self.dict else -2 for word in sentences.split(" ")] for sentences
                in before] #按照dict索引将文档进行编码
            for sentences in before_encode:
                if len(sentences) < self.max_length_word:
                    extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                    sentences.extend(extended_words)
            if len(before_encode) < self.max_length_sentences:
                extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                    range(self.max_length_sentences - len(before_encode))]
                before_encode.extend(extended_sentences)
            before_encode = [sentences[:self.max_length_word] for sentences in before_encode][
                            :self.max_length_sentences]
        else:
            for i in range(self.max_length_sentences):
                vector = [-1 for j in range(self.max_length_word)]
                before_encode.append(vector)
        before_encode = np.stack(arrays=before_encode, axis=0) 
        before_encode += 2
        #---------------------------------------------------------
        document_encode = [ 
            [self.dict.index(word) if word in self.dict else -2 for word in sentences.split(" ")] for sentences
            in text] #按照dict索引将文档进行编码
        #得到嵌套列表，将word映射到字典索引，若没有出现在dict中，则取-1

        #下面进行padding操作
        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences] 
        #如果文本长于所设定的max_length_word、max_length_sentences，则将多余的部分截取掉

        document_encode = np.stack(arrays=document_encode, axis=0) 
        #将嵌套列表转换为一个矩阵，max_length_sentences * max_length_word
        document_encode += 2
        # print(document_encode.shape)
        return document_encode.astype(np.int64), label, before_encode.astype(np.int64)

#测试主函数
if __name__ == '__main__':
    test = MyDataset(data_path="acl_data_train_new_around.csv", dict_path="glove.6B.100d.txt")
    # print(test.__getitem__(index=1)[0][1])