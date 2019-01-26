import numpy as np
from sklearn import svm

x = np.array([[1, 2],
              [5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11]])
y = [0, 1, 0, 1, 0, 1]

clf = svm.SVC(kernel='linear', C=2.0, probability=True)

clf.fit(x,y)
print(clf.predict([[0.58, 0.76]]))

print(clf.predict([[1.5, 1.8]]))

""" import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df.dropna(inplace=True)
df["Positive rated"] = np.where(df['sentiment'] > 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['Positive rated'], random_state=0)

vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
print('success')

X_train_vetorised = vect.transform(X_train)

clf = svm.SVC(kernel='linear', C=1.0, probability=True)
type(clf)
clf.fit(X_train_vetorised, y_train)

predictions = clf.predict(vect.transform(X_test))

print("AUC:SVM_Not_Cleaned", roc_auc_score(y_test, predictions))
testing = input("Enter the sentence for testing: ")

print(clf.predict(vect.transform([testing])))
print(clf.predict_proba(vect.transform([testing]))) """
