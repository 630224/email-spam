CODERSCAVE(EMAIL SPAM DETECTION)

PHASE 2 PROJECT (EMAIL SPAM DETECTION) NORMAL TASK

import numpy as np
import pandas as pd

data=pd.read_csv('/content/spam.csv')
data

data.columns

data.info()

data.isna().sum()

data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

clf.fit(X_train,y_train)

emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]

clf.predict(emails)

clf.score(X_test,y_test)
