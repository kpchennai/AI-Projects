import pandas as pd

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
print(test.head())

colsRes = ['class']

trainArr = train.values(cols)

trainRes = train.values(colsRes)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(trainArr, trainRes)

testArr = test.values(cols)

results = rf.predict(testArr)

test['predictions'] = results

