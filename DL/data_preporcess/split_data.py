import csv
import random
from sklearn.model_selection import train_test_split

def split_data(data_path, output_path, file_name):
    data_list, lable_list = [],[]
    with open(data_path, 'r', encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file) #将csv数据按行读下来
        rows = list(reader)
        csv_file.close()
    random.seed(123)
    # 打乱数据序列
    random.shuffle(rows)
    for row in rows:
        data_list.append([row[0]]+row[2:])
        lable_list.append(row[1])
        
    #先分为训练集和测试集
    train_X, test_X, train_Y, test_Y = train_test_split(data_list, lable_list, stratify=lable_list,
                                                        test_size=0.2, random_state=123)

    print(len(train_X))
    #再从训练集中分出验证集
    test_X, check_X, test_Y, check_Y = train_test_split(test_X, test_Y, stratify=test_Y,
                                                        test_size=0.5, random_state=666)
    print(len(test_X))
    print(len(check_X))

    filename = output_path+'\\'+file_name+'_train.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(train_X)):
            info = [train_X[i][0],train_Y[i]]+train_X[i][1:]
            csvwriter.writerow(info)
    filename = output_path+'\\'+file_name+'_test.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(test_X)):
            info = [test_X[i][0],test_Y[i]]+test_X[i][1:]
            csvwriter.writerow(info)
    filename = output_path+'\\'+file_name+'_valid.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(check_X)):
            info = [check_X[i][0],check_Y[i]]+check_X[i][1:]
            csvwriter.writerow(info)

if __name__ == '__main__':
    data_path = r"..\..\data\output\dl_file\around_1_content_with_title.csv"
    output_path = "..\..\data\output\dl_file"
    file_name = "around_1_content_with_title"
    split_data(data_path, output_path, file_name)
