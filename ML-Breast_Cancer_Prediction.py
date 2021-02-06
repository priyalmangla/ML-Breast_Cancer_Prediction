import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

df=pd.read_csv('data.csv') #dataset

'''Data Analysis'''
df.head() # check first five rows
# print(df.columns) # get all columns in dataset
df.info # identifying null columns from information of data set

# Dropping 'id' column and column that returns null.
df.drop('id' , axis=1, inplace=True)
# print(df.columns)
df.drop('Unnamed: 32' , axis=1, inplace=True)
# print(df.columns)

# Data Grouping 
# We can also group these to analyse more things, by clustering same features
l=list(df.columns) 
features_mean = l[1:11] 
features_se = l[11:21]
features_worst = l[21:]
# print(features_mean,  features_se, features_worst)

df['diagnosis'].unique() # M= Malignant, B= Benign


'''Data Exploration'''
df.describe() # summary of all the numeric columns
len(df.columns)
sns.countplot(df['diagnosis'], label="Count",); # will view bar graph of M and B cancer patients comparison 
df['diagnosis'].value_counts()
df.shape

# Correlation Plot
corr = df.corr()
corr.shape
plt.figure(figsize=(8,8))
sns.heatmap(corr);
# plt.show()

'''CREATE DATA FOR MACHINE LEARNING'''
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
df['diagnosis'].unique()
X = df.drop('diagnosis', axis=1)
# print(X.head())
y = df['diagnosis']
# print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # Train data and test data to be fed to learn, test size is 30%
df.shape
# Train data and test data count, we can verify from below that test size is 30% also
X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler() # Scaling all column values, Keeping a standard difference between high and low
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

'''Machine Learning Models'''

# Logic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score
lr_acc = accuracy_score(y_test, y_pred) # LR Accuracy

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
dtc_acc = accuracy_score(y_test, y_pred) # DTC Accuracy

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test) 
rfc_acc = accuracy_score(y_test, y_pred) # RFC Accuracy

# Support Vector Classifier
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, y_pred) # SVC Accuracy

# Compare all accuracies
# Accuracy score can vary on various factors, this result keeps on changing
results = pd.DataFrame({'Algorithm':['Logic Regression','Decision Tree Classifier','Random Forest Classifier','Support Vector Classifier Method'],
                        'Accuracy':[lr_acc,dtc_acc,rfc_acc,svc_acc]})
print(results)



