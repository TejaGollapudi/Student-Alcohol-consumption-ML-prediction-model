import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib





def test(X_train, X_test, y_train, y_test,a):
	
	if a==1:
		scaler = StandardScaler()
		scaler.fit(X_train)
		StandardScaler(copy=True, with_mean=True, with_std=True)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		model = joblib.load('mlp-model.sav')
		predictions= model.predict(X_test)
		print(classification_report(y_test,predictions))
		print(model.score(X_test,y_test))
	elif a==2:
		model = joblib.load('svm-model.sav')
		predictions= model.predict(X_test)
		print(classification_report(y_test,predictions))
		print(model.score(X_test,y_test))
	elif a==3:
		model = joblib.load('svm-model-ovr.sav')
		predictions= model.predict(X_test)
		print(classification_report(y_test,predictions))
		print(model.score(X_test,y_test))
		


df=pd.read_csv('mergedng-yn2binary-dummycoding.csv')



X=df.drop('Walc',axis=1)
X=df.drop('Dalc',axis=1)
Y=df['Walc']
X_train, X_test, y_train, y_test = train_test_split(X, Y)


x=0
while x==0:

	a=input('enter 1 to test mlp, enter 2 to test svm-ova,enter 3 to test svm-ovr');

	test(X_train, X_test, y_train, y_test,a)

	x=input('enter 0 to continue ');




