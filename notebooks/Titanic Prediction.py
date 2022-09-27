#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


# In[3]:


print(train.columns.values)


# In[4]:


train.info()


# In[5]:


train.tail(15)


# In[6]:


categorical_cols = [col for col in train.columns
                   if (train[col].dtype == 'object' or train[col].dtype == 'int64') and
                   train[col].nunique() < 10]
categorical_cols


# In[7]:


# How many passengers survived according to Pclass?
pclass_survived = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count()
pclass_survived['Percentage'] = round(pclass_survived['Survived'] / len(train) * 100, 2)
pclass_survived


# In[8]:


# How many passengers survived according to Age?
age_survived = (
    train[['Sex', 'Survived']]
    .groupby(['Sex'], as_index=False)
    .count()
    .sort_values(by='Survived', ascending=False)
)
age_survived['Percentage'] = round(age_survived['Survived'] / len(train) * 100, 2)
age_survived


# In[9]:


# How many passengers survived according to SibSp
sibsp_survived = (
    train[['SibSp', 'Survived']]
    .groupby(['SibSp'], as_index=False)
    .count()
    .sort_values(by='Survived', ascending=False)
)
sibsp_survived['Percentage'] = round(sibsp_survived['Survived'] / len(train) * 100, 2)
sibsp_survived


# In[10]:


# How many passengers survived according to ParCh
parch_survived = (
    train[['Parch', 'Survived']]
    .groupby(['Parch'], as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
)
parch_survived['Percentage'] = round(parch_survived['Survived'] / len(train) * 100, 2)
parch_survived


# In[11]:


# How many passengers survived according to place of embarkment?
embarkment_survived = (
    train[['Embarked', 'Survived']]
    .groupby(['Embarked'], as_index=False)
    .mean()
    .sort_values(by='Survived', ascending=False)
)
embarkment_survived['Percentage'] = round(embarkment_survived['Survived'] / len(train) * 100, 2)
embarkment_survived


# In[12]:


pclass_age_grid = sns.FacetGrid(train, col='Survived', row='Pclass', aspect=1.6)
pclass_age_grid.map(plt.hist, 'Age', bins=15)


# In[13]:


line_plt = sns.FacetGrid(train, row='Embarked', aspect=1.6)
line_plt.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
line_plt.add_legend()


# In[14]:


fare_pclass_grid = sns.FacetGrid(train, aspect=1.6)
fare_pclass_grid.map(sns.barplot, 'Pclass', 'Fare', alpha=.5, ci=None)

# Since there is a linear correlation between the fare and the Pclass, we can discard the fare


# In[15]:


train.head()


# ## Handle missing values of relevant columns

# In[16]:


train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[17]:


test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(test['Title'], test['Sex'])


# In[18]:


# Combine redundant titles
train['Title'] = train['Title'].replace(['Ms'], 'Miss')
test['Title'] = test['Title'].replace(['Ms'], 'Miss')


# In[19]:


# Handle missing Age values in train
mean_age_per_title = train[['Title', 'Age']].groupby(['Title'], as_index=False).mean().round(1)
dict_age_per_title = mean_age_per_title.to_dict('index')
dict_age_per_title


# In[20]:


train[train['Title'] == 'Mr'].head()


# In[21]:


for rec in range(0,len(dict_age_per_title.keys())):
    train.loc[train['Title'] == dict_age_per_title[rec]['Title'], 'Age'] = (
        train
        .loc[train['Title'] == dict_age_per_title[rec]['Title'], 'Age']
        .fillna(dict_age_per_title[rec]['Age'])
    )

train[train['Title'] == 'Mr'].head()


# In[22]:


# Handle missing Age values in test
mean_age_per_title = test[['Title', 'Age']].groupby(['Title'], as_index=False).mean().round(1)
dict_age_per_title = mean_age_per_title.to_dict('index')
dict_age_per_title


# In[23]:


for rec in range(0,len(dict_age_per_title.keys())):
    test.loc[test['Title'] == dict_age_per_title[rec]['Title'], 'Age'] = (
        test
        .loc[test['Title'] == dict_age_per_title[rec]['Title'], 'Age']
        .fillna(dict_age_per_title[rec]['Age'])
    )


# In[24]:


test[test['Title'] == 'Mr'].head()


# In[ ]:





# ## Feature Engineering

# In[25]:


other_male_titles = ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir']
other_female_miss = ['Lady', 'Mlle']
other_female_mrs = ['Countess', 'Mme', 'Dona']

train['Title'] = train['Title'].replace(other_male_titles, 'Mr')
train['Title'] = train['Title'].replace(other_female_miss, 'Miss')
train['Title'] = train['Title'].replace(other_female_mrs, 'Mrs')

test['Title'] = test['Title'].replace(other_male_titles, 'Mr')
test['Title'] = test['Title'].replace(other_female_miss, 'Miss')
test['Title'] = test['Title'].replace(other_female_mrs, 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


title_label_encoder = LabelEncoder()

train['Title'] = title_label_encoder.fit_transform(train['Title'])
train.head()


# In[28]:


test['Title'] = title_label_encoder.transform(test['Title'])
test.head()


# In[29]:


# Drop irrelevant columns
drop_cols = ['Name', 'Ticket', 'Fare', 'Cabin']

train = train.drop(drop_cols, axis=1)
test = test.drop(drop_cols, axis=1)


# In[30]:


train_y = train['Survived']
train = train.drop(['Survived'], axis=1)

train.shape, test.shape


# In[31]:


sex_label_encoder = LabelEncoder()
train['Sex'] = sex_label_encoder.fit_transform(train['Sex'])
train.head()


# In[32]:


test['Sex'] = sex_label_encoder.transform(test['Sex'])
test.head()


# In[33]:


train.info()


# In[34]:


train[train['Embarked'].isnull()]


# In[56]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")
data = imputer.fit_transform(train['Embarked'].values.reshape(-1,1)).ravel()

train['Embarked'] = data


# In[57]:


embarked_label_encoder = LabelEncoder()

train['Embarked'] = embarked_label_encoder.fit_transform(train['Embarked'])


# In[58]:


test['Embarked'] = embarked_label_encoder.transform(test['Embarked'])
test.head()


# In[59]:


train.shape, train_y.shape, test.shape


# ## Evaluation

# In[71]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split


# In[89]:


X_train, X_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=20)


# In[90]:


X_train = X_train.drop(['PassengerId'], axis=1)
X_valid = X_valid.drop(['PassengerId'], axis=1)


# In[91]:


def evaluate_model(model):
    model.fit(X_train, y_train)
    
    pred_score = round(model.score(X_valid, y_valid) * 100, 2)
    return pred_score


# In[92]:


models = {
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(),
    'LinearSVC': LinearSVC(),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
    'GaussianNB': GaussianNB(),
    'Perceptron': Perceptron(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'SGDClassifier': SGDClassifier()
}

for key in list(models.keys()):
    prediction_score = evaluate_model(models.get(key))
    
    print('Model: ', key, ' Score: ', prediction_score)


# In[ ]:




