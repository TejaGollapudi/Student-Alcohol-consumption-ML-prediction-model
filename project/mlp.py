import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
df=pd.read_csv('mergedng-yn2binary-dummycoding.csv')
#print(df.describe().transpose())
#print(df.head())

X=df.drop('Walc',axis=1)
X=df.drop('Dalc',axis=1)

Y=df['Walc']
#print(X.describe().transpose())
#print(Y.describe().transpose())

####data split############################################
X_train, X_test, y_train, y_test = train_test_split(X, Y)
##########################################################
#print(X_train)
#################normalisation of data####################
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print(X_train)

mlp = MLPClassifier(hidden_layer_sizes=(41),max_iter=2000,solver='lbfgs',tol=1e-10,activation='tanh',alpha=0.01)
mlp.fit(X_train,y_train)



predictions = mlp.predict(X_test)
print(classification_report(y_test,predictions))
print(mlp.score(X_test,y_test))

####################cross validation######################
clf = MLPClassifier(hidden_layer_sizes=(41),max_iter=2000,solver='lbfgs',tol=1e-15,activation='tanh')

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores2= cross_val_score(clf, X, Y, cv=5)

#cross validation without custom parameters##############
#scores = cross_val_score(clf, X, Y, cv=5)
print(scores2)

#########################################################



#################to save model###########################

"""
x=input()
if (x==1):
	joblib.dump(mlp,'mlp-model.sav')
	print('saved')

"""

########################################################