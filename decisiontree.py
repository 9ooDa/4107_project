# Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics

df = pd.read_csv('german_credit_data.csv')
# Data preparation
df['Sex'] = df['Sex'].replace(['male'],0)
df['Sex'] = df['Sex'].replace(['female'],1)
df['Housing'] = df['Housing'].replace(['own'],0)
df['Housing'] = df['Housing'].replace(['free'],1)
df['Housing'] = df['Housing'].replace(['rent'],2)
df['Saving accounts'] = df['Saving accounts'].fillna(0)
df['Saving accounts'] = df['Saving accounts'].replace(['little'],1)
df['Saving accounts'] = df['Saving accounts'].replace(['moderate'],2)
df['Saving accounts'] = df['Saving accounts'].replace(['quite rich'],3)
df['Saving accounts'] = df['Saving accounts'].replace(['rich'],4)
df['Checking account'] = df['Checking account'].fillna(0)
df['Checking account'] = df['Checking account'].replace(['little'],1)
df['Checking account'] = df['Checking account'].replace(['moderate'],2)
df['Checking account'] = df['Checking account'].replace(['quite rich'],3)
df['Checking account'] = df['Checking account'].replace(['rich'],4)
df['Risk'] = df['Risk'].replace(['good'],1)
df['Risk'] = df['Risk'].replace(['bad'],0)

dataset = df.values
X = dataset[:,1:9]
Y = dataset[:,10]
# print('X',X)
# print('Y',Y)
X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scale = min_max_scaler.fit_transform(X)
# X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# fit final model
model = DecisionTreeClassifier(criterion="entropy", max_depth=100000)
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
y_sum = 0
for ind in range(len(Y_test)):
    y_sum += Y_test[ind]
y_mean = y_sum / len(Y_test)
ssr = 0
sst = 0
ynew = model.predict(X_test)
# for i in range(len(X_test)):
#     print("X= {}, True_Y= {} ,Predicted= {}".format(X_test[i], Y_test[i] ,ynew[i]))
#     ssr += (Y_test[i] - ynew[i])**2
#     sst += (Y_test[i] - y_mean)**2
# r2 = 1 - (ssr/sst)
# print("R^2 value:",r2)
# print("r2_score:",metrics.r2_score(Y_test, ynew))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 2)
print(accuracies.mean())
print(accuracies.std())

