import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from spam import SpamDetector

data = pd.read_csv("spam.csv", encoding="latin-1")
nb = MultinomialNB()

data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1": "label", "v2": "text"})
print(data.head())
tags = data["label"]
texts = data["text"]

X, y = texts, tags

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

vectorizer = CountVectorizer()
X_train_dtm = vectorizer.fit_transform(X_train)
print(X_train_dtm)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
sd = SpamDetector()
sd.fit(list(X_train_dtm), list(y_train))

X_test_dtm = vectorizer.transform(X_test)
y_pred_class = nb.predict(X_test_dtm)
s_y_pred_class = sd.predict(X_test_dtm.toarray())

print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.accuracy_score(y_test, s_y_pred_class))
