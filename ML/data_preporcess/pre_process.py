#对原始语料进行分句、分词、去停用词、词型还原的处理
import csv
import nltk
import re
import os

def senttokenize(string):#分句代码     
    sent=nltk.sent_tokenize(string)#分句，存储成列表(用nltk分句函数)
    for line in sent[1:]:#从第二句开始判断是否出现错误
        if line[0:4]=='????':#特殊情况处理
            continue
        if (not line[0].istitle()): #是否首字母是大写
            k=sent[1:].index(line)#原句子元组里索引首字母不是大写的那句话的位置
            k=k+1#从第二句开始算，比实际位置少1，所以要加1 
            present=sent[k-1]#line的前面一句
            present=present+line
            sent[k-1]=present
            sent.pop(k)#当一句话的首字母不是大写时，把它加到前面一句，并删除这一句

        if re.search(r'^S\d+',line):#开头是数字
            k=sent[1:].index(line)
            k=k+1
            present=sent[k-1]
            if re.search(r'Fig\.',present):
                present=present+line
                sent[k-1]=present
                sent.pop(k)#如果是Fig.1被切开，当前句子并到前一句

        if re.search(r'^[A-Z]-?[A-Z]?\.$',line) :#or re.search(r'^\d+\.',line) or re.search(r'e\.g\.$',line)
            k=sent[1:].index(line)
            k=k+1
            present=sent[k-1]
            present=present+line
            sent[k-1]=present
            sent.pop(k)#标题被切开

        if re.search(r'^M\.I\.T',line):
            k=sent[1:].index(line)
            k=k+1
            present=sent[k-1]
            if re.search(r'Farrington\.',present):
                present=present+line
                sent[k-1]=present
                sent.pop(k)#人名被切开

    for s in sent[1:]:
        word_num=len(re.split(r' ',s)) #nltk.word_tokenize(s)
        if ' ' not in s or word_num<4:
            k=sent[1:].index(s)
            k=k+1
            present=sent[k-1]
            present=present+s
            sent[k-1]=present
            sent.pop(k)#最后整理，排除长度小于4的句子       
    return sent

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from nltk import PorterStemmer

abbre_list = ["e.g.","et al.","etc.","i.e."]
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#加载停用词表
with open('../../data/stopwords.txt', encoding="utf-8") as f:
    stopword_list = f.read().splitlines()
f.close()
nltk_stopword = stopwords.words('english')
stopword_list.extend(nltk_stopword)
stopword_list = list(set(stopword_list))
punctuation = '’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~ '
# punctuation_English = '!"#$%&()*+,/:;<=>?@[\]^_`{|}~'

with open('../../data/output/ACL_articles_data.csv', 'r', encoding="utf-8-sig") as f:
# file = open('C:\\Users\\诸葛绝才\\Desktop\\ACL全文章节抽取.csv', 'r')
    reader = csv.reader(f)
    rows = list(reader)
    f.close()

filename = '../../data/output/ACL_articles_preprocess.csv'
with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
    csvwriter = csv.writer(write_file, dialect='excel')
    # headers = ["文章ID","类型","内容"]
    # csvwriter.writerow(headers)
    for row in rows:
        #章节标题处理
        title_text = row[2].strip()
        title_tokens = nltk.word_tokenize(title_text.lower())
        title_filter = []
        for each_title_tokens in title_tokens:
            if (re.search("^\d+\.?$",each_title_tokens)==None) and (each_title_tokens not in punctuation):
                title_filter.append(each_title_tokens)
        title = " ".join(title_filter)
        #章节文本处理
        para = row[3].strip()
        for remove in abbre_list:
            para = para.replace(remove,"")
        para = para.replace(" '"," ")
        split_word = nltk.word_tokenize(para.lower())
        tokens_filter = [i for i in split_word if i not in punctuation]
        #剔除包含标点符号的字符串
        tokens = []
        for each_filter in tokens_filter:
            sign = re.search('[a-z]', each_filter)
            if (re.search("[!\"#$%&()*+,/:;<=>?@[\]^_`{|}~]",each_filter)==None) and (each_filter[0] not in punctuation) and (each_filter[-1] not in punctuation) and (sign!=None):
                tokens.append(each_filter)
        #分词处理
        # tokens = nltk.word_tokenize(sentence.lower())
        # tokens_filter = [i for i in tokens if i not in punctuation]
        # tokens = [nltk.word_tokenize(wait_divide.lower())]#分词
        #去停用词
        size_filter = [i for i in tokens if len(i)>2] 
        stopwords_filter = [i for i in size_filter if i not in stopword_list]
        #词干提取
        # porter = PorterStemmer()
        # pStems = [porter.stem(i) for i in number_filter]
        #词形还原
        lemmas_para = []
        wnl = WordNetLemmatizer()#词性标注句柄
        tagged_sent = nltk.pos_tag(stopwords_filter)#打词性标签
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_para.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  
        number_filter = [i for i in lemmas_para if (re.match("^\d+$",i)==None) and (re.match("^\d+[a-zA-Z]?$",i)==None) and (re.match("^\d+\.\d+$",i)==None) and (re.match("\d+-\d+",i)==None)]
        # final_words_list.extend(number_filter)
        final_string = " ".join(number_filter)
        #结果写入csv文件 
        info = [row[0],row[1],title,final_string]
        csvwriter.writerow(info)
        print(row[0]+"写入数据成功！")
print("完成！")