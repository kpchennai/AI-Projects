import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords

df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
df.dropna(inplace=True)
df["Positive rated"] = np.where(df['sentiment'] > 0, 1, 0)


def cleaning_words(raw_words):
    exam = BeautifulSoup(raw_words, "html.parser")  # removing html tags
    letters = re.sub("[^a-zA-Z]", " ",
                     exam.get_text())  # removing numbers and others except small and capital alphabets
    low = letters.lower()  # Converting everything to lower case
    words = low.split()  # spiliting sentences into words
    useful = [w for w in words if not w in stopwords.words("english")]  # removing stopping words
    use_sent = " ".join(useful)
    return use_sent


num = df["review"].size
# print num
perfect_words = []

for i in range(0, num):
    # if( (i+1)%1000 == 0 ):
    print("Review %d of %d\n" % (i + 1, num))
    print(cleaning_words(df["review"][i]))
    perfect_words.append(cleaning_words(df["review"][i]))

# print df.head(67)
# print df["Postive rated"].mean()
X_train, X_test, y_train, y_test = train_test_split(perfect_words, df['Positive rated'], random_state=0)
# print X_train[10]
# print X_train.shape
# print df['Postive rated']

vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
print(len(vect.get_feature_names()))

X_train_vetorised = vect.transform(X_train)
# print X_train_vetorised

print("starting training!!!!!")

model = LogisticRegression()
print("Stage 1 is completed")

model.fit(X_train_vetorised, y_train)
print("Stage 2 is completed")

# svm
# model_svm = SVC(kernel='linear', C = 1.0,probability=True)

# model_svm.fit(X_train_vetorised,y_train)
#

predictions = model.predict(vect.transform(X_test))
print("Stage 3 is completed")

print("AUC:LS_Cleaned", roc_auc_score(y_test, predictions))

#
# predictions=model_svm.predict(vect.transform(X_test))

# Sprint ("AUC:SVM_cleaned:",roc_auc_score(y_test,predictions))
#

feature_name = np.array(vect.get_feature_names())
sort_coeff = model.coef_[0].argsort()
print("small coeff : {}", format(feature_name[sort_coeff[:10]]))
print("large coeff : {}", format(feature_name[sort_coeff[:-11:-1]]))

testing = input("Enter the sentence for testing: ")
# print(model_svm.predict(vect.transform([testing])))
# print(model_svm.predict_proba(vect.transform([testing])))

print(model.predict(vect.transform([testing])))
print(model.predict_proba(vect.transform([testing])))
