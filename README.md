# 文献结构功能识别

## 项目简介
* 随着全文本学术文献的不断丰富，大量的研究工作关注于对文献知识的挖掘与组织。为了更好地实现对文献中知识要素的抽取与挖掘，了解文献的结构功能特征非常重要。
* 本研究着眼于文献章节粒度的结构功能识别工作，分别采用传统机器学习与基于神经网络的方法来构建分类模型，并且在特征输入上进行优化方式的探索，并基于不同的特征输入，对神经网络的模型结构进行适应性的调整优化。

## 基于传统机器学习的方法
* 特征选择：
  * 文本特征：章节标题信息（title）、章节内容信息（content）
  * 非文本特征：章节引用数特征（citation）、章节图表数之和特征（ft）、章节相对位置特征（loc）
* 文本特征预处理方案：
  * 去除停用词
  * 词形还原
  * 卡方特征选择
  * 特征值TF-IDF
* 模型选择：LR（线性回归）、NB（朴素贝叶斯）、KNN（K最近邻）、SVM（支持向量机）
* 特征优化方案：
  * input：content+title
  * input：content+citation+ft+loc
  * input：content+title+content+citation+ft+loc

## 基于神经网络的方法
* 特征选择：
  * 文本特征：章节标题内容（title）、章节文本内容（content）
  * 上下文特征
    * 上下文章节标题/文本信息，窗口大小1（around1）
    * 上下文章节标题/文本信息，窗口大小2（around2）
    * 上下文章节标题/文本信息，窗口大小3（around3）
* 文本特征预处理方案：
  * 不做额外特征过滤
  * Glove 100d Embedding
* 模型选择：basic模型选择（Bi-LSTM、Hierarchical Networks、Hierarchical Networks+Attention）
* 特征优化方案：
  * input：content+around1
  * input：content+around2
  * input：content+around3
  * input：title+around1
  * input：title+around2
  * input：title+around3
  * input：content+title+around1
  * input：content+title+around2
  * input：content+title+around3

## 项目结构
    ├─data  数据文件夹
    │  ├─output 数据文件输出
    │  │  ├─content_chi_dict  章节文本卡方值字典
    │  │  ├─dl_file  基于神经网络方法的数据文件
    │  │  ├─dl_model_save  神经网络模型保存
    │  │  └─nonsemantic_feature  非语义特征数据文件
    │  └─sample_articles  抽样文章数据
    ├─DL  深度学习训练模型代码
    │  ├─around_1_content_with_title
    │  ├─around_1_content_with_title_based_on_cnn
    │  ├─around_2_content_with_title
    │  ├─around_2_content_with_title_based_on_cnn
    │  ├─around_3_content_with_title
    │  ├─around_3_content_with_title_based_on_cnn
    │  ├─around_content_1
    │  ├─around_content_1_half
    │  ├─around_content_2
    │  ├─around_content_2_half
    │  ├─around_content_3
    │  ├─around_content_3_half
    │  ├─around_title_1
    │  ├─around_title_1_half
    │  ├─around_title_2
    │  ├─around_title_2_half
    │  ├─around_title_3
    │  ├─around_title_3_based_on_cnn
    │  ├─around_title_3_half
    │  ├─basic_model
    │  └─data_preporcess  数据预处理
    ├─ML  传统机器学习训练代码
    │  ├─create_nonsemantic_feature  构建非语义特征
    │  ├─create_text_feature  构建文本型特征
    │  ├─data_file_parsing  文献数据解析
    │  ├─data_preporcess  文献数据预处理
    │  └─model_train  模型训练
    └─utils  其他工具

## 代码说明
* ML_model

输入数据  | 代码文件 | 输出数据 | 说明
 ----- | ----- | ----- | -----
data\sample_articles  | data_file_parsing\extract_section_info.py | data\output\ACL_articles_data.csv | 抽取文献信息
data\output\ACL_articles_data.csv  | data_preporcess\pre_process.py | data\output\ACL_articles_preprocess.csv | 预处理文献信息
data\output\ACL_articles_preprocess.csv | create_text_feature\CHI_calculate.py | data\output\CHI-40%-new.txt | 获取卡方值过滤词典
data\output\ACL_articles_preprocess.csv  | create_text_feature\tf_idf_calculate.py | data\output\tfidf-vector-content.csv | 获得章节内容文本向量
data\output\ACL_articles_preprocess.csv  | create_nonsemantic_feature\citation_feature.py | data\output\citation_feature.csv | 获取引用数特征
data\output\ACL_articles_preprocess.csv  | create_nonsemantic_feature\ft_feature.py | data\output\ft_feature.csv | 获取图表数特征
data\output\ACL_articles_preprocess.csv  | create_nonsemantic_feature\relative_position_feature.py | data\output\relative_position_feature.csv | 获取相对位置特征
data\output\citation_feature.csv  | create_nonsemantic_feature\create_random.py | data\output\nonsemantic_feature\citation-100.csv | 生成引用数特征向量
data\output\ft_feature.csv  | create_nonsemantic_feature\create_random.py | data\output\nonsemantic_feature\ft-100.csv | 生成图表数特征向量
data\output\relative_position_feature.csv  | create_nonsemantic_feature\create_random.py | data\output\nonsemantic_feature\loc-100.csv | 生成相对位置特征向量
文本特征+非语义特征  | model_train\train_classifier.py | \ | 训练模型（LR、NB、KNN）
文本特征+非语义特征  | model_train\train_svm_classifier.py | \ | 训练模型（SVM）

* DL_model

输入数据  | 代码文件 | 输出数据 | 说明
 ----- | ----- | ----- | -----
data\output\ACL_articles_data.csv  | data_preporcess\pre_process_network.py | data\output\ACL_articles_preprocess_network.csv | 预处理文献信息
data\output\ACL_articles_preprocess_network.csv  | data_preporcess\create_around_data.py | data\output\dl_file（神经网络各模型输入数据） | 生成输入模型的总csv数据
data\output\dl_file（各模型输入总csv文件）  | DL\data_preporcess\split_data.py | data\output\dl_file（输入模型的train、valid、test文件） | 生成数据划分后的csv文件
data\output\dl_file\\ .csv  | basic_model | model.pkl | 基础神经网络模型训练（Bi-LSTM、HAN、HAN+Attention、CNN）
data\output\dl_file\\ .csv  | around_content_(1/2/3) | model.pkl | 基于章节内容信息，融合不同窗口大小的上下文信息
data\output\dl_file\\ .csv  | around_content_(1/2/3)_half | model.pkl | 基于章节内容信息，融合不同窗口大小的上下文信息（前向章节或后向章节）
data\output\dl_file\\ .csv  | around_title_(1/2/3) | model.pkl | 基于章节标题信息，融合不同窗口大小的上下文信息
data\output\dl_file\\ .csv  | around_title_(1/2/3)_half | model.pkl | 基于章节标题信息，融合不同窗口大小的上下文信息（前向章节或后向章节）
data\output\dl_file\\ .csv  | around_title_3_based_on_cnn | model.pkl | 基于章节标题信息，融合窗口为3的上下文信息，并以cnn模型作为上下文信息的融合模型
data\output\dl_file\\ .csv  | around_(1/2/3)_content_with_title | model.pkl | 基于章节标题+内容信息，融合不同窗口大小的上下文信息
data\output\dl_file\\ .csv  | around_(1/2/3)_content_with_title_based_on_cnn | model.pkl | 基于章节标题+内容信息，融合不同窗口大小的上下文信息，并以cnn模型作为上下文信息的融合模型

## 运行环境
* python==3.8.10
* pytorch==1.9.0
* cuda==10.0.130
* cudatoolkit==10.2.89
* libsvm==3.25
* scikit-learn==0.24.2
* nltk==3.6.2
* prefetch-generator==1.0.1

## 运行说明
* glove 100d 文件链接 https://pan.baidu.com/s/1zcgfnqTl5uElMvUh6tYg_A (ybse)，下载后放在data\output\dl_file目录下即可。
* 运行train_classifier.py，默认为LR模型，可以通过设置116行的classifier参数来设置模型，训练默认采用五折交叉验证
* 神经网络的所有代码文件夹中，train_me.py为模型训练脚本，apply_model.py为模型测试脚本，分别运行，生成模型保存在data\output\dl_model_save
* 所有代码均需进入其所在的文件夹目录下 运行
* 运行ML\create_text_feature\tf_idf_calculate.py脚本时需注意，需参考 https://blog.csdn.net/weixin_30711917/article/details/95900602 对sklearn包中feature_exceration文件夹中的text.py脚本进行简单修改，以保证使用TfidfVectorizer函数时对字符串切分的正确。

## Citation
Please cite the following paper if you use this codes and dataset in your work.

>Bowen Ma, Chengzhi Zhang, Yuzhuo Wang, Sanhong Deng. Enhancing Identification of Structure Function of Academic Articles Using Contextual Information. Scientometrics, 2022, 127(2): 885–925. [[doi]](https://doi.org/10.1007/s11192-021-04225-1)  [[arXiv]] (http://arxiv.org/abs/2111.14110)
