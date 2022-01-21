import csv
import nltk
import re
import os
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from scipy import sparse
#导入预处理后的数据
rows = []
with open(r'../../data/output/ACL_articles_preprocess.csv', 'r', encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    rows = list(reader)
    f.close()

# 加载特征词
filename = r'../../data/output/CHI-40%-new.txt'
vocabulary = []
with open(filename, 'r', encoding='utf-8') as file_read:
    lines = file_read.readlines() # 整行读取数据
    for each_line in lines:
        if len(each_line)>0:
            vocabulary.append(each_line.strip().split("\t")[0])
print("特征词数量:"+str(len(vocabulary)))

#构建语料
corpus = []#嵌套列表
for row in rows:
    corpus.append(row[3].strip())

#计算TF-IDF值
vectorizer = TfidfVectorizer(norm="l2",
                            smooth_idf=True,
                            token_pattern=r'\s+')
tf_idf = vectorizer.fit_transform(corpus)
# word_id_sorted = sorted(vectorizer.vocabulary_.items(),key = lambda x:x[1])
# print(word_id_sorted[0])

# 特征词list
word_list = vectorizer.get_feature_names()

matrix = coo_matrix(tf_idf).tocsr()

nonzero = matrix.nonzero()
matrix_dict = {}
for i in range(0,len(rows)):
    alist = []
    for j in range(0,len(word_list)):
        alist.append(np.float(0))
    matrix_dict[i] = alist

p = -1 #列计数器
for i in nonzero[0]:#遍历行坐标
    p = p+1
    column = nonzero[1][p]#这个词的字典编号就是它属于第几列
    value = matrix[i,column]
    matrix_dict[i][column] = value

#输出到csv

filename = r'../../data/output/tfidf-vector-content.csv'
with open(filename, 'w', newline="") as write_file:
    csvwriter = csv.writer(write_file)
    for i in range(0,len(rows)):
        info = [rows[i][0],rows[i][1]]
        info.extend(matrix_dict[i])
        csvwriter.writerow(info)
print("完成！")