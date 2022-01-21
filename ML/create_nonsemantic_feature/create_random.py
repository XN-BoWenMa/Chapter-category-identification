import csv
import random
import numpy as np

dimension = 100 # 维数设置
random.seed(123)

# 生成图表数特征
def create_ft_feature():
    result = []
    ft_dict = {}
    with open(r'..\..\data\output\ft_feature.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        f.close()
    figure_fea, table_fea = [], []
    for row in rows:
        table_fea.append(row[2])  
        figure_fea.append(row[3])
    for i in range(0,5):
        random_list = []
        for j in range(0,dimension):
            random_list.append(random.randint(1,15)*0.01)
        ft_dict[i] = random_list
    # print(ft_dict)
    for i in range(0,len(rows)):
        num = int(figure_fea[i])+int(table_fea[i])
        if num == 0:
            result.append(ft_dict[0])
        elif num in range(1,3):
            result.append(ft_dict[1])
        elif num in range(3,5):
            result.append(ft_dict[2])
        elif num in range(5,7):
            result.append(ft_dict[3])
        else:
            result.append(ft_dict[4])

    filename = r'..\..\data\output\nonsemantic_feature\ft-100.csv'
    with open(filename, 'w', newline="") as write_file:
        csvwriter = csv.writer(write_file)
        for i in result:
            csvwriter.writerow(i)
    print("完成！")

# 生成引用数特征
def create_citation_feature():
    result = []
    cite_dict = {}
    with open(r'..\..\data\output\citation_feature.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        f.close()
    cite_fea = []
    for row in rows:
        cite_fea.append(row[2])
    for i in range(0,4):
        random_list = []
        for j in range(0,dimension):
            random_list.append(random.randint(1,15)*0.01)
        cite_dict[i] = random_list
    for i in range(0,len(rows)):
        num = int(cite_fea[i])
        if num == 0:
            result.append(cite_dict[0])
        elif num in range(1,4):
            result.append(cite_dict[1])
        elif num in range(4,9):
            result.append(cite_dict[2])
        else:
            result.append(cite_dict[3])

    filename = r'..\..\data\output\nonsemantic_feature\citation-100.csv'
    with open(filename, 'w', newline="") as write_file:
        csvwriter = csv.writer(write_file)
        for i in result:
            csvwriter.writerow(i)
    print("完成！")

# 生成相对位置特征
def create_loc_feature():
    result = []
    loc_dict = {}
    with open(r'..\..\data\output\relative_position_feature.csv', 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        f.close()
    loc_fea = []
    for row in rows:
        loc_fea.append(row[2])
    begin = 1
    end = 15
    for i in range(0,5):
        random_list = []
        for j in range(0,dimension):
            random_list.append(random.randint(begin,end)*0.01)  
        loc_dict[i] = random_list
    for i in range(0,len(rows)):
        num = float(loc_fea[i])
        if 0<=num<0.2:
            result.append(loc_dict[0])
        elif 0.2<=num<0.4:
            result.append(loc_dict[1])
        elif 0.4<=num<0.7:
            result.append(loc_dict[2])
        elif 0.7<=num<0.9:
            result.append(loc_dict[3])
        else:
            result.append(loc_dict[4])
    #输出数据
    filename = r'..\..\data\output\nonsemantic_feature\loc-100.csv'
    with open(filename, 'w', newline="") as write_file:
        csvwriter = csv.writer(write_file)
        for i in result:
            csvwriter.writerow(i)
    print("完成！")

if __name__ == '__main__':
    create_ft_feature()
    create_citation_feature()
    create_loc_feature()
