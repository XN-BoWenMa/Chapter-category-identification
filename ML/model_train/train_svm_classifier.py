import csv
import nltk
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from random import shuffle
from libsvm.svmutil import *
from libsvm.svm import *

def get_data():

    # 导入文本向量
    data = []
    labels = []
    label_dict = {"introduction":0,"method":1,"evaluation and result":2,"related work":3,"conclusion":4}
    path = "..//..//data//output//nonsemantic_feature//"
    # 加载数据
    loc = pd.read_csv(path+'loc-100.csv', header=None)
    cite = pd.read_csv(path+'citation-100.csv', header=None)
    ft = pd.read_csv(path+'ft-100.csv', header=None)
    df = pd.read_csv(r'..\..\data\output\tfidf-vector-content.csv', header=None)
    '''附加其他特征'''
    # df = pd.concat([df, loc], axis=1, ignore_index=True)
    # df = pd.concat([df, cite], axis=1, ignore_index=True)
    # df = pd.concat([df, ft], axis=1, ignore_index=True)
    # df = pd.concat([df, loc, cite], axis=1, ignore_index=True)
    # df = pd.concat([df, cite, ft], axis=1, ignore_index=True)
    # df = pd.concat([df, loc, ft], axis=1, ignore_index=True)
    df = pd.concat([df, loc, cite, ft], axis=1, ignore_index=True)
    # 打乱数据
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    labels = df[1].values
    for i in range(len(labels)):
        labels[i] = label_dict[labels[i]]
    vector = df.loc[:,2:].values
    
    print(len(vector[0]))
    #row[文章ID，章节类型，文本]
    return vector, labels.astype('int')

def prepare_datasets(data_list, lable_list, test_data_proportion):
    train_X, test_X, train_Y, test_Y = train_test_split(data_list, lable_list,test_size=test_data_proportion,
                                                        stratify=lable_list, random_state=66)
    return train_X, test_X, train_Y, test_Y

def get_metrics(true_labels, predicted_labels):
    label_dict = {0:"introduction",1:"method",2:"evaluation and result",3:"related work",4:"conclusion"}
    true_labels = [label_dict[i] for i in true_labels]
    predicted_labels = [label_dict[i] for i in predicted_labels]
    report = classification_report(true_labels, predicted_labels)
    return report

def main():
    data, labels = get_data()  # 获取数据集
 
    #5折交叉验证
    kf = KFold(n_splits=5,random_state=123,shuffle=True)
    p_scores = []
    r_scores = []
    f1_scores = []
    count = 0
    for train_index, test_index in kf.split(data):
        count += 1
        train_x, train_y = data[train_index,:], labels[train_index]
        test_x, test_y = data[test_index,:], labels[test_index]
        options = '-t 0 -h 0'#线性核函数，表现较好

        print("开始训练模型-折次"+str(count))
        model = svm_train(train_y,train_x,options)
        p_label, p_acc, p_val = svm_predict(test_y,test_x,model)
        label_dict = {0:"introduction",1:"method",2:"evaluation and result",3:"related work",4:"conclusion"}
        true_labels = [label_dict[i] for i in test_y]
        predicted_labels = [label_dict[i] for i in p_label]
        # report = classification_report(true_labels, predicted_labels)
        # print(report)
        p_scores.append(precision_score(test_y,p_label,average='macro'))
        r_scores.append(recall_score(test_y,p_label,average='macro'))
        f1_scores.append(f1_score(test_y,p_label,average='macro'))

    if len(p_scores)==5:
        print("准确率："+str(np.array(p_scores).mean())+"\t"+str(np.array(p_scores).max())+"\t"+str(np.array(p_scores).min()))
        print("召回率："+str(np.array(r_scores).mean())+"\t"+str(np.array(r_scores).max())+"\t"+str(np.array(r_scores).min()))
        print("f1值："+str(np.array(f1_scores).mean())+"\t"+str(np.array(f1_scores).max())+"\t"+str(np.array(f1_scores).min()))
    
    #保存模型
    # svm_save_model("svm_train_all_text_title.model", model)
if __name__ == "__main__":
    main()