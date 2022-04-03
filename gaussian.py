
# Gaussian NB
from sklearn.naive_bayes import GaussianNB
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
df['Purpose'] = df['Purpose'].replace(['radio/TV'],0)
df['Purpose'] = df['Purpose'].replace(['domestic appliances'],1)
df['Purpose'] = df['Purpose'].replace(['furniture/equipment'],2)
df['Purpose'] = df['Purpose'].replace(['repairs'],3)
df['Purpose'] = df['Purpose'].replace(['vacation/others'],4)
df['Purpose'] = df['Purpose'].replace(['car'],5)
df['Purpose'] = df['Purpose'].replace(['education'],6)
df['Purpose'] = df['Purpose'].replace(['business'],7)
df['Risk'] = df['Risk'].replace(['good'],1)
df['Risk'] = df['Risk'].replace(['bad'],0)

dataset = df.values
X = dataset[:,1:10]
Y = dataset[:,10]

X = np.asarray(X).astype('float32')
Y = np.asarray(Y).astype('float32')
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

model = GaussianNB()
Y_train = Y_train.astype('int')
model.fit(X_train, Y_train)
ynew = model.predict(X_test)

for i in range(len(X_test)):
    print("X= {}, True_Y= {} ,Predicted= {}".format(X_test[i], Y_test[i] ,ynew[i]))

print("Accuracy Score:", model.score(X_test,Y_test))