import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

class MyDataset(Dataset):

    def __init__(self, data_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        labels, around_before, titles = [], [], []
        label_sign = {"introduction":0,"method":1,"evaluation and result":2,"related work":3,"conclusion":4}
        with open(data_path, 'r', encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file) #将csv数据按行读下来
            rows = list(reader)
            csv_file.close()   
        for row in rows:
            label = int(label_sign[row[1]])
            titles.append(row[2].split(" "))
            if len(row[4])>0:
                around_before.append(row[4].split(" "))
            else:
                around_before.append("")
            labels.append(label)
        self.labels = labels
        self.titles = titles
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
        title = self.titles[index]
        title_encode = [self.dict.index(word) if word in self.dict else -2 for word in title]
        before = self.around_before[index]

        before_encode = []
        if len(before)>0:
            before_encode = [self.dict.index(word) if word in self.dict else -2 for word in before]
            if len(before_encode) < 4:
                extended_words = [-1 for _ in range(4-len(before_encode))]
                before_encode.extend(extended_words)
            else:
                before_encode = before_encode[:4]
        else:
            before_encode = [-1 for _ in range(4)]
        before_encode = [i+2 for i in before_encode]
        #--------------------------------------------------------

        if len(title_encode) < 4:
            extended_words = [-1 for _ in range(4-len(title_encode))]
            title_encode.extend(extended_words)
        else:
            title_encode = title_encode[:4]
        title_encode = [i+2 for i in title_encode]
        
        return label, np.array(title_encode).astype(np.int64), np.array(before_encode).astype(np.int64)

#测试主函数
if __name__ == '__main__':
    test = MyDataset(data_path="acl_data_train_new_around_title.csv", dict_path="glove.6B.100d.txt")
    # print(test.__getitem__(index=1)[0][1])