import csv
import nltk
import re
import os
import string
import numpy as np
import math
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn import preprocessing  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

'''
"introduction","method","evaluation and result","related work","conclusion"
目标：提取前n个卡方值作为文本的向量表示
1.首先是计算每一个类中的不同词的DF值（即：文档频率），以字典形式保存
2.然后遍历语料，在对应的语料标签类型下，提取A：即该类字典中该词DF值，B：在其他类型字典中提取该词DF值之和
C：在该类字典中找到没有包含该词的文档数，D：在其他类字典中统计没有包含该词的文档数之和
3.根据卡方公式，计算对应的卡方值（具体为某一词项在某一标签类别下的卡方值）
'''
introduction_dict, method_dict, evaluation_dict, related_dict, conclusions_dict = {},{},{},{},{}
introduction_list, method_list, evaluation_list, related_list, conclusions_list = [],[],[],[],[]
#导入预处理后的数据
with open('../../data/output/ACL_articles_preprocess.csv', 'r', encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    rows = list(reader)
    f.close()

N = len(rows)#总文档数（各类文档数之和）
#进行语料标签分类
for row in rows:
    if row[1]=="introduction":
        introduction_list.append(row[2].strip())
    elif row[1]=="method":
        method_list.append(row[2].strip())
    elif row[1]=="related work":
        related_list.append(row[2].strip())
    elif row[1]=="conclusion":
        conclusions_list.append(row[2].strip())
    else:
        evaluation_list.append(row[2].strip())

total_dict = [introduction_dict,method_dict,evaluation_dict,related_dict,conclusions_dict]
total_list = [introduction_list,method_list,evaluation_list,related_list,conclusions_list]
category = ["introduction","method","evaluation","related work","conclusion"]

#按类别分别构建DF字典 (文档计数字典)
for i in range(0,5):
    cv = CountVectorizer(token_pattern=r'\s+',binary=True)
    features = cv.fit_transform(total_list[i]) 
    word_id_dict = cv.vocabulary_
    # figure_id = word_id_dict['figure']
    length = len(cv.get_feature_names())
    word_list = cv.get_feature_names()
    df_list = list(features.toarray().sum(axis=0))#按列求和
    # print(category[i]+":"+str(df_list[figure_id]))
    for j in range(0,length):
        total_dict[i][word_list[j]] = df_list[j]
# print(total_dict[0])
# print("构建DF字典完成！")
# for i in total_dict:
#     print(len(i))

# 计算不同文本类别下词项的卡方值
chi_dict = []#保存各类文本卡方值字典

left_dict_list = []#做成一个嵌套字典
for i in range(0,5):
    left_dict = [total_dict[j] for j in range(0,5) if j!=i]
    left_dict_list.append(left_dict)

left_list_list = []
for i in range(0,5):
    left_list = [total_list[j] for j in range(0,5) if j!=i]
    left_list_list.append(left_list)

for i in range(0,5):
    each_chi_dict = {}
    for k,v in total_dict[i].items():
        if k not in each_chi_dict:
            value_A = np.float32(v)
            value_C = np.float32(len(total_list[i])-value_A)
            # left_dict = [total_dict[j] for j in range(0,5) if j!=i]
            value_B = np.float32(0)
            for each_left in left_dict_list[i]:
                if k in each_left:
                    value_B = value_B+each_left[k]
            value_D = np.float32(0)
            # left_list = [total_list[j] for j in range(0,5) if j!=i]
            for each_left in left_list_list[i]:
                value_D = value_D+len(each_left) 
            value_D = value_D-value_B
            value_CHI = N*(value_A*value_D-value_B*value_C)**2/((value_A+value_C)*(value_A+value_B)*(value_B+value_D)*(value_C+value_D))
            each_chi_dict[k] = value_CHI
    # chi_dict.append(each_chi_dict)
    
    pickle.dump(each_chi_dict, open("../../data/output/content_chi_dict/"+category[i]+'_dict.pkl','wb'))
    print(category[i]+"字典完成！")

# 取每个特征项的最大值来表示该词项特征
path = "../../data/output/content_chi_dict/"
introduction_chi_dict = pickle.load(open(path+'introduction_dict.pkl', 'rb'))
method_chi_dict = pickle.load(open(path+'method_dict.pkl', 'rb'))
evaluation_chi_dict = pickle.load(open(path+'evaluation_dict.pkl', 'rb'))
related_work_chi_dict = pickle.load(open(path+'related work_dict.pkl', 'rb'))
conclusions_chi_dict = pickle.load(open(path+'conclusion_dict.pkl', 'rb'))
dict_list = [introduction_chi_dict,method_chi_dict,evaluation_chi_dict,related_work_chi_dict,conclusions_chi_dict]
label_dict = {0:"introduction",1:"method",2:"evaluation and result",3:"related work",4:"conclusion"}
total_reslut = {}
for i in range(0,5):
    for k,v in dict_list[i].items():
        if k not in total_reslut:
            # total_reslut[k] = [v,label_dict[i]]
                total_reslut[k] = v
        else:
            # if v>total_reslut[k][0]:
            if v>total_reslut[k]:
                # total_reslut[k] = [v,label_dict[i]]
                total_reslut[k] = v
# 输出结果
final_result = {}
for k,v in total_reslut.items():
    # if v[1]=="conclusion":
    final_result[k] = v
final_result_sorted = sorted(final_result.items(),key = lambda x:x[1],reverse = True)
count = 0
for i in final_result_sorted:
    count += 1
    # print(i[0]+";"+str(i[1]))
    if count>=100:
        break

pickle.dump(total_reslut, open(path+'total_dict.pkl','wb'))
print("total字典完成！")

# 构建特征词项空间
path = "../../data/output/"
total_dict = pickle.load(open(r'..\..\data\output\content_chi_dict\total_dict.pkl', 'rb'))
# print(total_dict["figure"])
total_dict_sorted = sorted(total_dict.items(),key = lambda x:x[1],reverse = True)
stop_pos = int(len(total_dict_sorted)*0.4)
with open(path+"CHI-40%-new.txt","w",encoding="utf-8") as f:
    for i in range(0,stop_pos):
        f.write(total_dict_sorted[i][0]+"\t"+str(total_dict_sorted[i][1])+"\n")
print('content向量构建完成！')