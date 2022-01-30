---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# ML-with-python Capstone Project

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
```

To download dataset

```python
#!wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv
```

## Data description
This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

| Field          | Description                                                                           |
| -------------- | ------------------------------------------------------------------------------------- |
| Loan_status    | Whether a loan is paid off on in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |



## Data Cleansing and Preparation

- Data transformation from catagorical to numerical
- Rmoving unnecessary/unrelavent data or features

### Data Structure 

```python
df = pd.read_csv('loan_train.csv')
print(f'Shape of dataset : {df.shape}.\n')
print(f'Data types :\n{df.dtypes}\n')
print(f'Columns: \n{df.columns}')

```

```python
print(f'Data structure: \n')
df.head()
```

### Visualization of some features

```python
df['effective_date'] = pd.to_datetime(df['effective_date'])
df['effective_date'] = df['effective_date'].dt.dayofweek

bins = np.linspace(df.effective_date.min(), df.effective_date.max(),7)
#fig, axes = plt.subplots(2,2, figsize=(20,20))
gp = sns.FacetGrid(df, col='Gender',
                   hue='loan_status',
                   palette='Set1',
                   col_wrap=2, 
                   height=4,
                   aspect=1.0 )
gp.map(plt.hist, 'effective_date', bins=len(bins), ec='k')
#gp.figure.savefig('delet_1.png')
gp.axes[1].legend()
```

It seems, who got the loan begining of the week they are pay their loan off. Let's do feature binarization with a threshold value 3.

```python
df['due_date'] = pd.to_datetime(df['due_date'])
df['due_date'] = df['due_date'].dt.dayofweek

bins = np.linspace(df.due_date.min(), df.due_date.max(),7)
gp = sns.FacetGrid(df, col='Gender',
                   hue='loan_status',
                   palette='Set1',
                   col_wrap=2,
                   height=4,
                   aspect=1.0)

gp.map(plt.hist, 'due_date', bins=len(bins), ec='k')
gp.axes[-1].legend()
```

There is no such epecific pattern for **'loan_status'** regarding **'due_data'**. Thus we can skip this data column.


Only two data samples are member of 'Master or Above' subclass, Therefore we can exclude the this subclass from our dataset. Oneway to do it transform these subclass, e.g. **'college', Bechalor**, into featurs label. 

```python
df['Principal'].value_counts(),df['terms'].value_counts()
```

### Preparation of features and terget

```python
# 0-> for the first 4 days
# 1-> for the last 3 days
df['effective_date'] = df['effective_date'].apply(lambda x: 1 if (x>3)  else 0)
df.drop('due_date', axis=1, inplace=True)

Feature = df[['loan_status', 'Principal', 'terms', 'effective_date', 
              'age', 'education', 'Gender']]
# Removin the data sample with Master or Above

Feature = Feature[Feature['education']!='Master or Above']
Feature = Feature[Feature['Principal'] >= 800]
#Feature = Feature[Feature['terms'] > 7]
#Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
#Feature.drop(['Master or Above'], axis=1, inplace=True)

Feature['education'].replace(to_replace=['High School or Below', 
                                         'Bechalor', 'college'], 
                               value=[0,1,2], inplace=True)
le_gender = preprocessing.LabelEncoder().fit(['female', 'male'])
Feature['Gender']= le_gender.transform(Feature['Gender'])
Feature['loan_status'].replace(to_replace=['PAIDOFF', 'COLLECTION'], 
                               value=[1,0], inplace=True)
X = Feature.drop(['loan_status'], axis=1)
y = Feature['loan_status']
```

```python
Feature.head()
```

## Data Normalization and Train-test spliting

```python
from sklearn.model_selection import train_test_split
X[['Principal', 'terms', 'age',]] = preprocessing.StandardScaler().fit(X[['Principal','terms', 'age',]]
                                                                      ).transform(X[['Principal', 'terms', 'age',]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,)
```

## Classification Algorithm
### K-Nearest Neighbour

```python
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import jaccard_score, f1_score, log_loss, confusion_matrix
```

```python
k = range(20)[3:]
jac_metrics = []
log_metrics = []
f1_metrics = []
for i in k: 
    
    knc = KNeighborsClassifier(n_neighbors=i, p=2)
    knc.fit(X_train, y_train)
    y_hat = knc.predict(X_test)
    
    ja = jaccard_score(y_true=y_test, y_pred=y_hat)
    jac_metrics.append(ja)
    ll = 1-log_loss(y_true=y_test, y_pred=y_hat) /len(y_test)
    log_metrics.append(ll)
    f1 = f1_score(y_true=y_test, y_pred=y_hat)
    f1_metrics.append(f1)
```

```python
k_best = []
k_best.append(k[jac_metrics.index(max(jac_metrics))])
k_best.append(k[log_metrics.index(max(log_metrics))])
k_best.append(k[f1_metrics.index(max(f1_metrics))])
k_best = max(k_best, key = k_best.count)

plt.figure(figsize=(8,8))
plt.plot(k, jac_metrics, 'g', label='Jaccard index')
plt.plot(k, log_metrics, 'b', label='1-log_loss')
plt.plot(k, f1_metrics, 'r', label='f1 prcession')
plt.xlabel('neighbour number(k)')
plt.ylabel('accuracy')
plt.title('k-Nearest Neighbour Algorithm')
plt.axvline(k_best, ls=':', alpha=0.5, color='gray', label='Optimum k value')
plt.legend()
```

#### Calclating the confusion matrix

```python
knc = KNeighborsClassifier(n_neighbors=k_best, p=2)
knc.fit(X_train, y_train)
y_hat = knc.predict(X_test)

Conf_matr = np.zeros(shape=(2,2), dtype=int)
for (i,j) in zip(y_test, y_hat):
    if i==0:
        if j==0:
            Conf_matr[0,0] = Conf_matr[0, 0] + 1
        else:
            Conf_matr[0,1] = Conf_matr[0, 1] + 1
    else:
        if j==0: 
            Conf_matr[1,0] = Conf_matr[1,0] + 1
        else:
            Conf_matr[1,1] = Conf_matr[1,1] + 1
#Conf_matr==confusion_matrix(y_true=y_test, y_pred=y_hat)
```

```python
ja = jaccard_score(y_true=y_test, y_pred=y_hat)
ll = 1-log_loss(y_true=y_test, y_pred=y_hat) /len(y_test)
f1 = f1_score(y_true=y_test, y_pred=y_hat)

print(' Metrics for KNeighborsClassifier --')
print(' Jaccard score : ', ja)
print(' 1-logloss : ', ll)
print(' F1 score : ', f1)
```

```python
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

```python
plot_confusion_matrix(cm=Conf_matr,
                      title='Confusion matrix with K-Nearest Neighbour',
                       classes=['COLLECTION(0)', 'PAIDOFF(1)'])
```

### Decision Tree Algorithm

```python
from sklearn.tree import DecisionTreeClassifier
criterion = 'gini'#'entropy'
dtc = DecisionTreeClassifier(criterion=criterion)
dtc.fit(X_train, y_train)
y_hat = dtc.predict(X_test)

```

```python
ja = jaccard_score(y_true=y_test, y_pred=y_hat)
ll = 1-log_loss(y_true=y_test, y_pred=y_hat) /len(y_test)
f1 = f1_score(y_true=y_test, y_pred=y_hat)

print(' Metrics for Decision Tree Algorithm--')
print(' Jaccard score : ', ja)
print(' 1-logloss : ', ll)
print(' F1 score : ', f1)
```

```python
y_test.value_counts()
```

#### Calclating the confusion matrix

```python
Conf_matr = confusion_matrix(y_true=y_test, 
                 y_pred=y_hat)

plot_confusion_matrix(cm=Conf_matr,
                      title='Confusion matrix with Decision Tree',
                      classes=['COLLECTION(0)', 'PAIDOFF(1)'])
```

### Support Vector Machine Algorithm


#### With rbf kernel

```python
from sklearn import svm
svm_al = svm.SVC(kernel='rbf', gamma='scale')
svm_al.fit(X_train, y_train)

y_hat = svm_al.predict(X_test)
```

```python
ja = jaccard_score(y_true=y_test, y_pred=y_hat)
ll = 1-log_loss(y_true=y_test, y_pred=y_hat) /len(y_test)
f1 = f1_score(y_true=y_test, y_pred=y_hat)

print(' Metrics for Support Vector Machine with rbf kernel --')
print(' Jaccard score : ', ja)
print(' 1-logloss : ', ll)
print(' F1 score : ', f1)
```

#### Calclating the confusion matrix

```python
Conf_matr = confusion_matrix(y_true=y_test, 
                 y_pred=y_hat)

plot_confusion_matrix(cm=Conf_matr,
                      title='Confusion matrix with Support Vector Machine (rbf)',
                      classes=['COLLECTION(0)', 'PAIDOFF(1)'])
```

#### With polynomial (poly) kernel

```python
#{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
from sklearn import svm
svm_al = svm.SVC(kernel='poly', degree=9)
svm_al.fit(X_train, y_train)

y_hat = svm_al.predict(X_test)
```

```python
ja = jaccard_score(y_true=y_test, y_pred=y_hat)
ll = 1-log_loss(y_true=y_test, y_pred=y_hat) /len(y_test)
f1 = f1_score(y_true=y_test, y_pred=y_hat)

print(' Metrics for Support Vector Machine with poly kernel --')
print(' Jaccard score : ', ja)
print(' 1-logloss : ', ll)
print(' F1 score : ', f1)
```

```python
Conf_matr = confusion_matrix(y_true=y_test, 
                 y_pred=y_hat)

plot_confusion_matrix(cm=Conf_matr,
                      title='Confusion matrix with Support Vecctor Machine (poly)',
                      classes=['COLLECTION(0)', 'PAIDOFF(1)'])
```

### Clssification with Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lr = LogisticRegression(C=0.1, solver='liblinear').fit(X_train,y_train)

y_hat = lr.predict(X_test)
```

```python
ja = jaccard_score(y_true=y_test, y_pred=y_hat)
ll = 1-log_loss(y_true=y_test, y_pred=y_hat) /len(y_test)
f1 = f1_score(y_true=y_test, y_pred=y_hat)

print(' Metrics for Support Vector Machine with poly kernel --')
print(' Jaccard score : ', ja)
print(' 1-logloss : ', ll)
print(' F1 score : ', f1)
```

```python
Conf_matr = confusion_matrix(y_true=y_test, 
                 y_pred=y_hat)

plot_confusion_matrix(cm=Conf_matr,
                      title='Confusion matrix with Logistic regression',
                      classes=['COLLECTION(0)', 'PAIDOFF(1)'])
```

## Observation and Conclusion
- Data cleansing and preparation is important part of ML as it improves the results. For example removing two samples corresponds to the `education[Master or Above`] and five samples associated with `Principle[<800]` improves the confusion matrix, though the changes in the accuracy is not significant.
- The metrics discussed here give almost the same accuracy over different attempts.
- Significant changes in confusion matrix have observed over the different attempts, e.g. on some attempts the confusion matrix gives better result in both `PAIDOFF(1)` and `COLLECION(0)` and on other attempts better result comes only in qestion of `PAIDOFF`.
- Regarding the overall performances, the algorithm Decision Tree renders better prediction.  
- Cross-validation might be a resovent for point(3) which will improve the parameters( $\large{\theta}$ from the expression $\large{\theta}^\intercal\large{x}$ ) implementing algorithm over the attempts. 
