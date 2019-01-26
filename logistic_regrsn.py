import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df.dropna(inplace=True)
df["Positive rated"] = np.where(df['sentiment'] > 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['Positive rated'], random_state=0)

vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)

X_train_vetorised = vect.transform(X_train)

model = LogisticRegression()
print("Logistic regression training going on")

model.fit(X_train_vetorised, y_train)

print("Logistic regression training completed")
# svm
print("SVM training going on")

model2_svm = svm.SVC(kernel='linear', C=1.0, probability=True)

model2_svm.fit(X_train_vetorised, y_train)

print("SVM training Completed")
#

predictions = model.predict(vect.transform(X_test))

print("AUC:LS_not_Cleaned", roc_auc_score(y_test, predictions))
#
predictions = model2_svm.predict(vect.transform(X_test))

print("AUC:SVM_Not_Cleaned", roc_auc_score(y_test, predictions))
#
feature_name = np.array(vect.get_feature_names())

sort_coeff = model.coef_[0].argsort()

print("small coeff : {}", format(feature_name[sort_coeff[:10]]))

print("large coeff : {}", format(feature_name[sort_coeff[:-11:-1]]))

testing = input("Enter the sentence for testing: ")

print(model2_svm.predict(vect.transform([testing])))
print(model2_svm.predict_proba(vect.transform([testing])))

print(model.predict(vect.transform([testing])))
print(model.predict_proba(vect.transform([testing])))
