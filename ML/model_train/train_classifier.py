import csv
import nltk
import re
import os
import xlrd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
from matplotlib.pylab import style
import random

def get_data():

    # 导入文本向量
    data = []
    labels = []
    label_dict = {"introduction":0,"method":1,"evaluation and result":2,"related work":3,"conclusion":4}
    # 加载数据
    path = "..//..//data//output//nonsemantic_feature//"
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

def train_predict_evaluate(classifier,
                           data, labels):
    # 模型训练-十折交叉验证
    kf = KFold(n_splits=5, random_state=666, shuffle=True)
    p_scores = []
    r_scores = []
    f1_scores = []
    count = 0
    for train_index, test_index in kf.split(data):
        count += 1
        print("开始训练模型-折次"+str(count))
        train_x, train_y = data[train_index,:], labels[train_index]
        test_x, test_y = data[test_index,:], labels[test_index]
        classifier.fit(train_x, train_y)
        # joblib.dump(classifier, "knn_best_model_with_feature_new.m")
        predictions = classifier.predict(test_x)
        label_dict = {0:"introduction",1:"method",2:"evaluation and result",3:"related work",4:"conclusion"}
        true_labels = [label_dict[i] for i in test_y]
        predicted_labels = [label_dict[i] for i in predictions]
        report = classification_report(true_labels, predicted_labels, digits=4)
        # print(str(metrics.confusion_matrix(true_labels, predicted_labels)))
        # print(report)
        p_scores.append(precision_score(test_y,predictions,average='macro'))
        r_scores.append(recall_score(test_y,predictions,average='macro'))
        f1_scores.append(f1_score(test_y,predictions,average='macro'))
    if len(p_scores)==5:
        print("准确率："+str(np.array(p_scores).mean())+"\t"+str(np.array(p_scores).max())+"\t"+str(np.array(p_scores).min()))
        print("召回率："+str(np.array(r_scores).mean())+"\t"+str(np.array(r_scores).max())+"\t"+str(np.array(r_scores).min()))
        print("f1值："+str(np.array(f1_scores).mean())+"\t"+str(np.array(f1_scores).max())+"\t"+str(np.array(f1_scores).min()))
    
    # 模型保存
    # path = "C:\\Users\\诸葛绝才\\Desktop\\"
    # classifier.fit(data, labels)
    # joblib.dump(classifier, path+"lr_train_all_model_text_title.m")

def main():
    data, labels = get_data()  # 获取数据集

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    # 创建模型对象
    mnb = MultinomialNB()
    lr = LogisticRegression(multi_class="multinomial", max_iter=10000)#,max_iter=10000
    knn = KNeighborsClassifier(n_neighbors=7)
    svm = SVC(kernel='linear',decision_function_shape='ovo')

    # 模型训练
    model_predictions = train_predict_evaluate(classifier=lr,
                                               data=data,
                                               labels=labels)                                      
    
    #网格搜索调参
    # lr = LogisticRegression()
    # param_grid = {"penalty": ["l1","l2"],
    #               "C": [0.1,0.3,0.5,0.7,1.0],
    #               "multi_class": ["multinomial","ovr"]}
                  
    # grid_search = GridSearchCV(lr, param_grid=param_grid, n_jobs=4, cv=10, scoring='f1_macro')
    # grid_search.fit(train_data, train_labels)
    # print(grid_search.grid_scores_)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    # params = {'max_depth':range(3, 7), 'n_estimators':range(100, 1100, 200), 'learning_rate':[1.5, 2.0, 2.5]}
    # xgbc_best = XGBClassifier()
    # gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
    # gs.fit(X_train, y_train)

    #保存模型
    # joblib.dump(mnb_tfidf_predictions, filename="MultinomialNB.pkl")

if __name__ == "__main__":
    main()