import csv

path = "..\\..\\data\\output\\"
with open(path+'ACL_articles_preprocess_network.csv', 'r', encoding="utf-8-sig") as f1:
    reader = csv.reader(f1)
    rows1 = list(reader)
    f1.close()

def create_around_1_title():
    # 前后一章节
    output_path = "..\\..\\data\\output\\dl_file\\"
    current_id = "" #设置初始文章id
    filename = output_path+'around_title_1.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(rows1)):
            paper_id = rows1[i][0]
            if paper_id==current_id:
                if (i+1)<len(rows1) and (rows1[i+1][0] == current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][2],rows1[i+1][2]]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][2],'']              
            else:
                current_id = paper_id
                if rows1[i+1][0] == current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'',rows1[i+1][2]]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'','']
            csvwriter.writerow(info)

def create_around_1_content():
    output_path = "..\\..\\data\\output\\dl_file\\"
    current_id = "" #设置初始文章id
    filename = output_path+'around_content_1.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(rows1)):
            paper_id = rows1[i][0]
            if paper_id==current_id:
                if (i+1)<len(rows1) and (rows1[i+1][0] == current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][3],rows1[i+1][3]]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][3],'']              
            else:
                current_id = paper_id
                if rows1[i+1][0] == current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'',rows1[i+1][3]]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'','']
            csvwriter.writerow(info)

def create_around_1_title_half(sign):
    # 前后一章节_half
    output_path = "..\\..\\data\\output\\dl_file\\"
    current_id = "" #设置初始文章id
    if sign=='before':
        filename = output_path+'around_title_1_before.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id==current_id:
                    if (i+1)<len(rows1) and (rows1[i+1][0] == current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][2]]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'']               
                else:
                    current_id = paper_id
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][2]]
                csvwriter.writerow(info)
    elif sign=='after':
        filename = output_path+'around_content_1_after.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id==current_id:
                    if (i+1)<len(rows1) and (rows1[i+1][0] == current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2]]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'']               
                else:
                    current_id = paper_id
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2]]
                csvwriter.writerow(info)

def create_around_1_content_half(sign):
    # 前后一章节_half
    output_path = "..\\..\\data\\output\\dl_file\\"
    current_id = "" #设置初始文章id
    if sign=='before':
        filename = output_path+'around_content_1_before.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id==current_id:
                    if (i+1)<len(rows1) and (rows1[i+1][0] == current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][3]]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'']               
                else:
                    current_id = paper_id
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-1][3]]
                csvwriter.writerow(info)
    elif sign=='after':
        filename = output_path+'around_content_1_after.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id==current_id:
                    if (i+1)<len(rows1) and (rows1[i+1][0] == current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3]]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],'']               
                else:
                    current_id = paper_id
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3]]
                csvwriter.writerow(info)

def create_around_2_title():
    # 前后两章节-title
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    filename = output_path+'around_title_2.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(rows1)):
            paper_id = rows1[i][0]
            if paper_id!=current_id:
                current_id = paper_id
                index = 1
            else:
                index += 1
            if index==1:
                if rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i+1][2],rows1[i+2][2]]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i+1][2],""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",""]
            elif index==2:
                if rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][2],rows1[i+1][2],rows1[i+2][2]]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][2],rows1[i+1][2],""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][2],"",""]
            elif index>2:
                if (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],rows1[i+2][2]]
                elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][2],rows1[i-1][2],"",""]
            csvwriter.writerow(info)

def create_around_2_content():
    # 前后两章节
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    filename = output_path+'around_content_2.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(rows1)):
            paper_id = rows1[i][0]
            if paper_id!=current_id:
                current_id = paper_id
                index = 1
            else:
                index += 1
            if index==1:
                if rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i+1][3],rows1[i+2][3]]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i+1][3],""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",""]
            elif index==2:
                if rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][3],rows1[i+1][3],rows1[i+2][3]]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][3],rows1[i+1][3],""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][3],"",""]
            elif index>2:
                if (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],rows1[i+2][3]]
                elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][3],rows1[i-1][3],"",""]
            csvwriter.writerow(info)

def create_around_2_title_half(sign):
    # 前后两章节_half
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    if sign=='before':
        filename = output_path+'around_content_2_before.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                elif index==2:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][2]]
                elif index>2:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][2],rows1[i-1][2]]
                csvwriter.writerow(info)
    elif sign=='after':
        filename = output_path+'around_content_2_after.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    if rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2]]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                elif index==2:
                    if rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2]]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                elif index>3:
                    if (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2]]
                    elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                csvwriter.writerow(info)

def create_around_2_content_half(sign):
    # 前后两章节_half
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    if sign=='before':
        filename = output_path+'around_content_2_before.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                elif index==2:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-1][3]]
                elif index>2:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-2][3],rows1[i-1][3]]
                csvwriter.writerow(info)
    elif sign=='after':
        filename = output_path+'around_content_2_after.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    if rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3]]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                elif index==2:
                    if rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3]]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                elif index>3:
                    if (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3]]
                    elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",""]
                csvwriter.writerow(info)

def create_around_3_title():
    # 前后三章节 title
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    filename = output_path+'around_title_3.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(rows1)):
            paper_id = rows1[i][0]
            if paper_id!=current_id:
                current_id = paper_id
                index = 1
            else:
                index += 1
            if index==1:
                if rows1[i+3][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                elif rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",rows1[i+1][2],rows1[i+2][2],""]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",rows1[i+1][2],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","","","",""]
            elif index==2:
                if rows1[i+3][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][2],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                elif rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][2],rows1[i+1][2],rows1[i+2][2],""]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][2],rows1[i+1][2],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][2],"","",""]
            elif index==3:
                if (i+3)<len(rows1) and rows1[i+3][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                elif (i+2)<len(rows1) and rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],rows1[i+2][2],""]
                elif (i+1)<len(rows1) and rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],"",rows1[i-2][1],rows1[i-1][1],"","",""]
            elif index>3:
                if (i+3)<len(rows1) and (rows1[i+3][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][2],rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                elif (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][2],rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],rows1[i+2][2],""]
                elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][2],rows1[i-2][2],rows1[i-1][2],rows1[i+1][2],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][2],rows1[i-2][2],rows1[i-1][2],"","",""]
            csvwriter.writerow(info)

def create_around_3_content():
    # 前后三章节 content
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    filename = output_path+'around_content_3.csv'
    with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
        csvwriter = csv.writer(write_file, dialect='excel')
        for i in range(len(rows1)):
            paper_id = rows1[i][0]
            if paper_id!=current_id:
                current_id = paper_id
                index = 1
            else:
                index += 1
            if index==1:
                if rows1[i+3][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                elif rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",rows1[i+1][3],rows1[i+2][3],""]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","",rows1[i+1][3],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","","","","",""]
            elif index==2:
                if rows1[i+3][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                elif rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][3],rows1[i+1][3],rows1[i+2][3],""]
                elif rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][3],rows1[i+1][3],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][3],"","",""]
            elif index==3:
                if (i+3)<len(rows1) and rows1[i+3][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                elif (i+2)<len(rows1) and rows1[i+2][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],rows1[i+2][3],""]
                elif (i+1)<len(rows1) and rows1[i+1][0]==current_id:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],"",rows1[i-2][1],rows1[i-1][1],"","",""]
            elif index>3:
                if (i+3)<len(rows1) and (rows1[i+3][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][3],rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                elif (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][3],rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],rows1[i+2][3],""]
                elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][3],rows1[i-2][3],rows1[i-1][3],rows1[i+1][3],"",""]
                else:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][3],rows1[i-2][3],rows1[i-1][3],"","",""]
            csvwriter.writerow(info)

def create_around_3_title_half(sign):
    # 前后三章节title_half
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    if sign=='before':
        filename = output_path+'around_title_3_before.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index==2:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][2]]
                elif index==3:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][2],rows1[i-1][2]]
                elif index>3:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][2],rows1[i-2][2],rows1[i-1][2]]
                csvwriter.writerow(info)
    elif sign=='after':
        filename = output_path+'around_title_3_after.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    if rows1[i+3][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                    elif rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],""]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index==2:
                    if rows1[i+3][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                    elif rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],""]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index==3:
                    if rows1[i+3][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                    elif rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],""]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index>3:
                    if (i+3)<len(rows1) and (rows1[i+3][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],rows1[i+3][2]]
                    elif (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],rows1[i+2][2],""]
                    elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][2],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                csvwriter.writerow(info)

def create_around_3_content_half(sign):
    # 前后三章节content_half
    output_path = "..\\..\\data\\output\\dl_file\\"
    index = 0
    current_id = ''
    if sign=='before':
        filename = output_path+'around_content_3_before.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index==2:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",rows1[i-1][3]]
                elif index==3:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"",rows1[i-2][3],rows1[i-1][3]]
                elif index>3:
                    info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i-3][3],rows1[i-2][3],rows1[i-1][3]]
                csvwriter.writerow(info)
    elif sign=='after':
        filename = output_path+'around_content_3_after.csv'
        with open(filename, 'w', newline="", encoding="utf-8-sig") as write_file:
            csvwriter = csv.writer(write_file, dialect='excel')
            for i in range(len(rows1)):
                paper_id = rows1[i][0]
                if paper_id!=current_id:
                    current_id = paper_id
                    index = 1
                else:
                    index += 1
                if index==1:
                    if rows1[i+3][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                    elif rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],""]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index==2:
                    if rows1[i+3][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                    elif rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],""]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index==3:
                    if rows1[i+3][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                    elif rows1[i+2][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],""]
                    elif rows1[i+1][0]==current_id:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                elif index>3:
                    if (i+3)<len(rows1) and (rows1[i+3][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],rows1[i+3][3]]
                    elif (i+2)<len(rows1) and (rows1[i+2][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],rows1[i+2][3],""]
                    elif (i+1)<len(rows1) and (rows1[i+1][0]==current_id):
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],rows1[i+1][3],"",""]
                    else:
                        info = [rows1[i][0],rows1[i][1],rows1[i][2],rows1[i][3],"","",""]
                csvwriter.writerow(info)

if __name__ == '__main__':
    # around1
    create_around_1_title()
    create_around_1_content()
    create_around_1_title_half('before')
    create_around_1_title_half('after')
    create_around_1_content_half('before')
    create_around_1_content_half('after')
    # around2
    create_around_2_title()
    create_around_2_content()
    create_around_2_title_half('before')
    create_around_2_title_half('after')
    create_around_2_content_half('before')
    create_around_2_content_half('after')
    # around3
    create_around_3_title()
    create_around_3_content()
    create_around_3_title_half('before')
    create_around_3_title_half('after')
    create_around_3_content_half('before')
    create_around_3_content_half('after')
