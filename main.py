import numpy as np
import pandas as pd
import re
from __init_opList import get_opList
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

pd.options.mode.chained_assignment = None

# 导入训练、测试数据集
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 对训练集与测试集合的text列进行词优化
opList = get_opList()
for data in opList:
    train_df['text'][:] = [re.sub(data, '', text) for text in train_df['text']]
    test_df['text'][:] = [re.sub(data, '', text) for text in test_df['text']]


# 训练、测试集向量化
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])
print(train_vectors[0])

# 提取标签
train_label = train_df['target']

# 依据训练集合划分测试集和向量集合，并构建朴素贝叶斯拟合模型
X_train, X_test, y_train, y_test \
    = train_test_split(train_vectors, train_label, test_size=0.1, random_state=7)
clf = BernoulliNB()
clf.fit(train_vectors, train_label)

# 使用测试数据集测试，并生成结果
y_pred = clf.predict_proba(test_vectors)
y_final = [int(y_pred[i][0] < y_pred[i][1]) for i in range(0, np.array(y_pred).shape[0])]


# 保存到提交文件中
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["target"] = y_final
sample_submission.to_csv("submission.csv", index=False)

print('Done.')

"""
acc = accuracy_score(y_test, y_final) #准确率
print("Accuracy: %2f%%" % (acc * 100.0))
f1 = sklearn.metrics.f1_score(y_test, y_final)  # F1分数
print("f1_score: %.2f%%" % (f1 * 100.0))
print(classification_report(y_test, y_final))  # 形成报表
"""


