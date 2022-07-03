# Identification of Structure Function of Academic Articles

## Background
With the increasing enrichment of full-text academic literature, a large number of research focus on the extraction and organization of knowledge elements in the literature. In order to better achieve the extraction and mining of knowledge elements, it is important to understand the structure function of academic articles.

  This project focuses on the Identification of structure function of academic articles in chapter granularity, and uses traditional machine learning and deep learning methods to train classifier respectively. We explore the optimization approach for feature inputs, and adjust the model structure of neural networks based on different feature inputs adaptively.

## Traditional Machine Learning Methods
* Feature selection: 
  * Text features: Chapter titles、Chapter contents
  * Non-semantic features：The number of citations (citation), Number of figures and tables (ft), Relative position of chapter (loc)
* Data preprocess: 
  * Removing stop words
  * Lemmatization
  * Chi-square feature selection
  * Vector feature weight: TF-IDF
* Model selection: Logistic Regression (LR), Naive Bayes (NB), K-nearest Neighbor (KNN), Support Vector Machine (SVM)
* Optimization of Feature Input: 
  * input: content + title
  * input: content + citation + ft + loc
  * input: content + title + content + citation + ft + loc

## Deep Learning Methods
* Feature selection:
  * Text features: Chapter titles、Chapter contents
  * Contextual Features:
    * Contextual chapter title/content information, window size is set to 1 (around1)
    * Contextual chapter title/content information, window size is set to 2 (around2)
    * Contextual chapter title/content information, window size is set to 3 (around3)
* Data preprocess: 
  * No additional feature filtering
  * Glove 100d Embedding
* Model selection: Base Model Selection (Bi-LSTM, Hierarchical Networks, Hierarchical Networks + Attention)
* Optimization of Feature Input: 
  * input: content + around1
  * input: content + around2
  * input: content + around3
  * input: title + around1
  * input: title + around2
  * input: title + around3
  * input: content + title + around1
  * input: content + title + around2
  * input: content + title + around3

## Project Structure

    Chapter-category-identification
    ├─data  (data folder)
    │  ├─output (output files folder)
    │  │  ├─content_chi_dict  (the chi-square value dictionary of Chapter content)
    │  │  ├─dl_file  (data files based on neural network methods)
    │  │  ├─dl_model_save  (neural network model save folder)
    │  │  └─nonsemantic_feature  (non-semantic feature files)
    │  └─sample_articles  (sample articles folder)
    ├─DL  (the codes of deep learning training)
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
    │  └─data_preporcess (data preprocessing)
    ├─ML  (the codes of traditional machine learning training)
    │  ├─create_nonsemantic_feature  (build non-semantic features)
    │  ├─create_text_feature  (build textual features)
    │  ├─data_file_parsing  (literature data analysis)
    │  ├─data_preporcess  (literature data preprocessing)
    │  └─model_train  (training models)
    └─utils  (other tools)

## Codes Description
* ML_model

<style>
table th:first-of-type {
	width: 100px;
}
</style>

data input  | code files | data output | description
 ----- | ----- | ----- | -----
data\sample_articles  | data_file_parsing\extract_section_info.py | data\output\ACL_articles_data.csv | extracting the information of articles
data\output\ACL_articles_data.csv  | data_preporcess\pre_process.py | data\output\ACL_articles_preprocess.csv | preprocessing text information
data\output\ACL_articles_preprocess.csv | create_text_feature\CHI_calculate.py | data\output\CHI-40%-new.txt | acquiring chi-square value dictionary
data\output\ACL_articles_preprocess.csv  | create_text_feature\tf_idf_calculate.py | data\output\tfidf-vector-content.csv | acquiring text vectors of chapter content
data\output\ACL_articles_preprocess.csv  | create_nonsemantic_feature\citation_feature.py | data\output\citation_feature.csv | acquiring the feature of citation
data\output\ACL_articles_preprocess.csv  | create_nonsemantic_feature\ft_feature.py | data\output\ft_feature.csv | acquiring the feature of ft
data\output\ACL_articles_preprocess.csv  | create_nonsemantic_feature\relative_position_feature.py | data\output\relative_position_feature.csv | acquiring the feature of loc
data\output\citation_feature.csv  | create_nonsemantic_feature\create_random.py | data\output\nonsemantic_feature\citation-100.csv | generating feature vectors of citation
data\output\ft_feature.csv  | create_nonsemantic_feature\create_random.py | data\output\nonsemantic_feature\ft-100.csv | generating feature vectors of ft
data\output\relative_position_feature.csv  | create_nonsemantic_feature\create_random.py | data\output\nonsemantic_feature\loc-100.csv | generating feature vectors of loc
contextual feature+non-semantic feature | model_train\train_classifier.py | \ | training model(LR、NB、KNN)
contextual feature+non-semantic feature  | model_train\train_svm_classifier.py | \ | training model(SVM)

* DL_model

data input  | code files | data output | description
 ----- | ----- | ----- | -----
data\output\ACL_articles_data.csv  | data_preporcess\pre_process_network.py | data\output\ACL_articles_preprocess_network.csv | preprocessing text information
data\output\ACL_articles_preprocess_network.csv  | data_preporcess\create_around_data.py | data\output\dl_file (Input data of deep learning models) | generating total input file(.csv)
data\output\dl_file (total input file for each model) | DL\data_preporcess\split_data.py | data\output\dl_file (training validing and testing files) | generating data files after data division(.csv)
data\output\dl_file\\ .csv  | basic_model | model.pkl | base neural network model training (Bi-LSTM, HAN, HAN+Attention, CNN)
data\output\dl_file\\ .csv  | around_content_(1/2/3) | model.pkl | based on chapter content, fusing contextual information with different window sizes
data\output\dl_file\\ .csv  | around_content_(1/2/3)_half | model.pkl | based on chapter content, fusing contextual information with different window sizes (forward chapters or backward chapters)
data\output\dl_file\\ .csv  | around_title_(1/2/3) | model.pkl | based on chapter title, fusing contextual information with different window sizes
data\output\dl_file\\ .csv  | around_title_(1/2/3)_half | model.pkl | based on chapter title, fusing contextual information with different window sizes (forward chapters or backward chapters)
data\output\dl_file\\ .csv  | around_title_3_based_on_cnn | model.pkl | based on the chapter title, the fusion window size of contextual information is set to 3, and the cnn model is adopted as the fusion model
data\output\dl_file\\ .csv  | around_(1/2/3)_content_with_title | model.pkl | based on chapter title and content，fusing contextual information with different window sizes
data\output\dl_file\\ .csv  | around_(1/2/3)_content_with_title_based_on_cnn | model.pkl | based on chapter title and content，fusing contextual information with different window sizes, and the cnn model is adopted as the fusion model

## Operating Environment
* python==3.8.10
* pytorch==1.9.0
* cuda==10.0.130
* cudatoolkit==10.2.89
* libsvm==3.25
* scikit-learn==0.24.2
* nltk==3.6.2
* prefetch-generator==1.0.1

## Operating Instructions
* glove 100d Link https://pan.baidu.com/s/1zcgfnqTl5uElMvUh6tYg_A (ybse) After downloading, put the file in the fold (data\output\dl_file).
* Run train_classifier.py, the default is the LR model, you can change the model by setting the classifier parameter in line 116, and the training result defaults to five-fold cross-validation.
* In all the code folders of the neural network, train_me.py is the model training script, apply_model.py is the model testing script, please run separately, and the generated model is saved in data\output\dl_model_save.
* All scripts should be run in the folder where it is located.
* Note when running the ML\create_text_feature\tf_idf_calculate.py script, you need to refer to https://blog.csdn.net/weixin_30711917/article/details/95900602 to make a simple modification to the text.py script in site-packages\sklearn\feature_extraction to ensure correct string slicing when using the TfidfVectorizer function.

## Citation
Please cite the following paper if you use this code and dataset in your work.

>Bowen Ma, Chengzhi Zhang, Yuzhuo Wang, Sanhong Deng. Enhancing Identification of Structure Function of Academic Articles Using Contextual Information. Scientometrics, 2022, 127(2): 885–925. [[doi]](https://doi.org/10.1007/s11192-021-04225-1)  [[arXiv]](http://arxiv.org/abs/2111.14110)
