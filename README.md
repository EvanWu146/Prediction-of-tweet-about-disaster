# README

## 问题描述

1. 小组选取的题目是Kaggle上的Natural Language Processing with Disaster Tweets(Predict which Tweets are about real disasters and which ones are not)，即通过自然语言处理NLP，来根据给定推文中的内容预测哪些内容说的是真实的灾害，而哪些不是。

2. 问题的形式化描述

   Kaggle上给了三个文件，分别是test.csv、train.csv和sample_submission.csv，我们需要通过train.csv中的7613条数据，其中包括id列、关键词列、位置列、文字列、target列（0、1二值实现二分类，0代表不是真实自然灾害，1代表是真实自然灾害）来训练数据模型，最终预测test.csv中的3263条推文内容是否分别和真实灾害的发生有关。最终将预测结果生成.csv文件后提交到Kaggle上。

## 系统

1. 系统架构

   本次任务使用了sklearn中的朴素贝叶斯分类器：

   ```python
   class sklearn.naive_bayes.BernoulliNB(*, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
   ```

   朴素贝叶斯分类器用于多元伯努利模型的朴素贝叶斯分类器。和 MultinomialNB 一样，这个分类器适用于离散数据，不同之处在于 MultinomialNB 适用于出现次数，而 BernoulliNB 设计用于二进制，即结果只有0和1的模型预测。

   我们在程序中先对数据集进行词优化，随后将训练集和测试集向量化，再使用朴素贝叶斯拟合模型进行训练，最后生成target列生成submission.csv文件提交至Kaggle项目的Ranking中。

2. 各个部分介绍

   1. 导入训练、测试数据集：使用pandas模块，实现对csv文件的读取并以dataframe形式存储数据集。
   2. 训练、测试集向量化：将数据集中的‘text‘列（即tweet内容列）向量化并分别存储于训练向量集和测试向量集。
   3. 提取标签：提取训练集中的’target‘列
   4. 依据训练集合划分测试集和向量集合，并构建朴素贝叶斯拟合模型：使用训练向量集生成模型
   5. 使用测试数据集测试，并生成结果：使用测试向量集进行预测
   6. 保存到提交文件中

3. 算法的伪代码

   1. 引用库：

   ```python
   import numpy as np
   import pandas as pd
   import re
   from __init_opList import get_opList
   from sklearn import feature_extraction
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import BernoulliNB
   ```
   
   2. 代码主体部分：(省略了部分写法，可执行代码main.py在项目文件夹里)
   
   ```python
   # 导入训练、测试数据集
   train_df, test_df= pd.read_csv('~')
   ```
   
   ```python
   # 对训练集与测试集合的text列进行词优化
   for data in opList = get_opList():
       (train_df, test_df)['text'][:] = [re.sub(data, '', text) for text in train_df['text']]
   ```
   
   ```python
   # 训练、测试集向量化
   train_vectors, test_vectors = \
   			feature_extraction.text.CountVectorizer().count_vectorizer.fit_transform((train_df, test_df)["text"]) 
   ```
   
   ```python
   # 提取标签
   train_label = train_df['target']
   ```
   
   ```python
   # 依据训练集合划分测试集和向量集合，并构建朴素贝叶斯拟合模型
   X_train, X_test, y_train, y_test \
       = train_test_split(train_vectors, ...)
   clf =  BernoulliNB()
   clf.fit(train_vectors, train_label)
   ```
   
   ```python
   # 使用测试数据集测试，并生成结果
   y_pred = clf.predict_proba(test_vectors)
   y_final = y_pred to [0, 1, ....]
   ```
   
   ```python
   # 保存到提交文件中
   sample_submission = pd.read_csv("sample_submission.csv")
   sample_submission["target"] = y_final
   sample_submission.to_csv("submission.csv", index=False)
   ```
   
   
