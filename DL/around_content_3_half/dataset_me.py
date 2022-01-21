import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels, around_before1, around_before2, around_before3 = [], [], [], [], [], [], [], []
        label_sign = {"introduction":0,"method":1,"evaluation and result":2,"related work":3,"conclusion":4}
        with open(data_path, 'r', encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file) #将csv数据按行读下来
            rows = list(reader)
            csv_file.close()
        for row in rows:
            text = row[3].split("\t")
            label = int(label_sign[row[1]])
            if len(row[4])>0:
                around_before3.append(row[4].split("\t"))
            else:
                around_before3.append("")
            if len(row[5])>0:
                around_before2.append(row[5].split("\t"))
            else:
                around_before2.append("")
            if len(row[6])>0:
                around_before1.append(row[6].split("\t"))
            else:
                around_before1.append("")
            texts.append(text)
            labels.append(label)

        self.texts = texts
        self.labels = labels
        self.around_before1 = around_before1
        self.around_before2 = around_before2
        self.around_before3 = around_before3
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
        before1 = self.around_before1[index]
        before2 = self.around_before2[index]
        before3 = self.around_before3[index]

        before_encode1 = []
        if len(before1)>0:
            before_encode1 = [ 
                [self.dict.index(word) if word in self.dict else -2 for word in sentences.split(" ")] for sentences
                in before1] #按照dict索引将文档进行编码
            for sentences in before_encode1:
                if len(sentences) < self.max_length_word:
                    extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                    sentences.extend(extended_words)
            if len(before_encode1) < self.max_length_sentences:
                extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                    range(self.max_length_sentences - len(before_encode1))]
                before_encode1.extend(extended_sentences)
            before_encode1 = [sentences[:self.max_length_word] for sentences in before_encode1][
                            :self.max_length_sentences]
        else:
            for i in range(self.max_length_sentences):
                vector = [-1 for j in range(self.max_length_word)]
                before_encode1.append(vector)
        before_encode1 = np.stack(arrays=before_encode1, axis=0) 
        before_encode1 += 2
        #--------------------------------------------------------
        before_encode2 = []
        if len(before2)>0:
            before_encode2 = [ 
                [self.dict.index(word) if word in self.dict else -2 for word in sentences.split(" ")] for sentences
                in before2] #按照dict索引将文档进行编码
            for sentences in before_encode2:
                if len(sentences) < self.max_length_word:
                    extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                    sentences.extend(extended_words)
            if len(before_encode2) < self.max_length_sentences:
                extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                    range(self.max_length_sentences - len(before_encode2))]
                before_encode2.extend(extended_sentences)
            before_encode2 = [sentences[:self.max_length_word] for sentences in before_encode2][
                            :self.max_length_sentences]
        else:
            for i in range(self.max_length_sentences):
                vector = [-1 for j in range(self.max_length_word)]
                before_encode2.append(vector)
        before_encode2 = np.stack(arrays=before_encode2, axis=0)
        before_encode2 += 2
        #--------------------------------------------------------
        before_encode3 = []
        if len(before3)>0:
            before_encode3 = [ 
                [self.dict.index(word) if word in self.dict else -2 for word in sentences.split(" ")] for sentences
                in before3] #按照dict索引将文档进行编码
            for sentences in before_encode3:
                if len(sentences) < self.max_length_word:
                    extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                    sentences.extend(extended_words)
            if len(before_encode3) < self.max_length_sentences:
                extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                    range(self.max_length_sentences - len(before_encode3))]
                before_encode3.extend(extended_sentences)
            before_encode3 = [sentences[:self.max_length_word] for sentences in before_encode3][
                            :self.max_length_sentences]
        else:
            for i in range(self.max_length_sentences):
                vector = [-1 for j in range(self.max_length_word)]
                before_encode3.append(vector)
        before_encode3 = np.stack(arrays=before_encode3, axis=0)
        before_encode3 += 2
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
        return document_encode.astype(np.int64), label, before_encode1.astype(np.int64), before_encode2.astype(np.int64), before_encode3.astype(np.int64)

#测试主函数
if __name__ == '__main__':
    test = MyDataset(data_path="acl_data_train_new_around.csv", dict_path="glove.6B.100d.txt")
    # print(test.__getitem__(index=1)[0][1])