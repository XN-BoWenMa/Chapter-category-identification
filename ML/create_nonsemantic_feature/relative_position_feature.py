import os
import re
import xml.dom.minidom
import xml.etree.ElementTree as et

import nltk
import xlwt
import csv
import codecs

def file_name(file_dir): #获取一个文件夹中子文件名列表 return list
    b = []
    for dirs in os.listdir(file_dir):
        b.append("..\\..\\data\\sample_articles\\"+dirs)
        # b.append("D:\\BaiduNetdiskDownload\\xml会议论文语料\\XML\\N(NAACL)\\"+dirs)
    return b

def senttokenize(string):#分句代码     
    sent=nltk.sent_tokenize(string)#分句，存储成列表(用nltk分句函数)
    for line in sent[1:]:
        if line[0:4]=='????':#特定规则
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
            sent.pop(k)#再整理，排除长度小于4的句子       
    return sent

filename = r'..\..\data\output\relative_position_feature.csv'
with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
    # write_file.write(codecs.BOM_UTF8) # 解决乱码问题
    csvwriter = csv.writer(write_file)
    file_total = file_name("..\..\data\sample_articles")
    file_total_length = len(file_total)
    limt = 0
    while limt<file_total_length:
        try:      
            file_name = file_total[limt].replace("..\\..\\data\\sample_articles\\",'')
            flie_ID = file_name[0:8]#获取文章编号
            # if flie_ID in useful_id:
            # if True:
            tree = et.parse(file_total[limt])
            root = tree.getroot()#生成根节点
            for i in root.getiterator("variant"):
                variant =  i
                break
            sections_list = list(variant)#element的序化列表
        #进行文档有效性的判断
            effective_section = ["introduction","related work","method",
            "evaluation","result","evaluation and result","conclusion","other"]
        #生成初始抽取字典
            length = len(sections_list)   
            list_sign = ["bodyText","listItem"]
            # list_sign = ["bodyText"]
            sec_extract = []#单篇文章文本保存
            for i in range(0,length,1):
                attri_dic = sections_list[i].attrib
                if sections_list[i].tag=="sectionHeader" and (attri_dic["genericHeader"] in effective_section):
                # if sections_list[i].tag=="sectionHeader" and (attri_dic["genericHeader"] != "references"):
                    section_title = sections_list[i].text.replace("\n"," ").replace("- ","").strip().lower()                          
                    q = i+1#进入下一个element
                    text = []#每个一级章节下的文本list
                    while sections_list[q].tag!="sectionHeader":
                        if sections_list[q].tag in list_sign:
                            text.append(sections_list[q].text.strip().replace("—","-").replace("–","-"))#将章节文本按每个标签取下来
                        if (q+1)<=(len(sections_list)-1):
                            q = q+1
                        else:
                            break
                    #保存一个章节的内容
                    sec_extract.append([attri_dic["genericHeader"],text])   
        except OSError:
            pass
            print("跳过"+flie_ID)
            
        #对章节文本进行整合
        effective_sec = []
        for each_sec in sec_extract:
            effective_sec.append(each_sec[0])
        effective_sec_length = len(effective_sec)
        effective_sec = list(set(effective_sec))
        if len(effective_sec)<3:
            limt = limt+1
            continue
        k = 0
        while k<len(sec_extract):
            para_list = []
            para_iter = sec_extract[k][1]
            if len(para_iter)>0:
                for each_part in para_iter:
                    each_part = each_part.replace("&quot;",'"').replace("&amp;","&").replace("&lt;","<").replace("&gt;",">").replace("","").replace("�","")
                    # 将文本分行去除
                    text_change = each_part.replace("\n"," ").replace("- ","")
                    text_change = text_change.replace("","")
                    # apart = re.findall('[\u0030-\u0039\u0041-\u005a\u0061-\u007a !"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]',each_part,re.S)#乱码字符串过滤
                    # exchange = "".join(apart)
                    para_list.append(text_change)
                para = " ".join(para_list)
                para = para.replace("- ","")
                sec_extract[k][1] = para
            else:
                sec_extract[k][1] = ""
            k = k+1
        for relative_loc in range(0,effective_sec_length):
            relative_value = (relative_loc+1)/effective_sec_length
            sec_extract[relative_loc].append(round(relative_value,2))
        # 写入csv文件
        four_section = ["introduction","related work","method","conclusion"]
        length = len(sec_extract)
        for i in range(0,length):
            if len(sec_extract[i][1])>0:
                if sec_extract[i][0] in four_section:
                    info = [flie_ID,sec_extract[i][0],sec_extract[i][2]]
                else:
                    info = [flie_ID,"evaluation and result",sec_extract[i][2]]
                csvwriter.writerow(info)          
            else:
                print(flie_ID+"异常")
               
        print(flie_ID+"写入数据成功！")
        limt = limt+1          
    #     break
    # break
print("抽取完成！")