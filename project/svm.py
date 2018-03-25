import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA


################load file#################################################
df=pd.read_csv('mergedng-yn2binary-dummycoding.csv')
#print(df.describe().transpose())
#print(df.head())



X=df.drop('Walc',axis=1)
X=df.drop('Dalc',axis=1)
X=df.drop('school_GP',axis=1)
X=df.drop('school_MS',axis=1)
Y=df['Walc']
#print(X.describe().transpose())
#print(Y.describe().transpose())


X_train, X_test, y_train, y_test = train_test_split(X, Y)

#print(X_train)

#print(X_train)

################PCA########################################
pca = PCA(.99)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train.shape)
print(X_train_pca.shape)

################################ovr rbf #################################

model = svm.SVC(decision_function_shape='ovo',max_iter=9000) 


###############linear svm model##########################################

"""
model=svm.LinearSVC(max_iter=90000)
"""
#########################################################################

model.fit(X_train, y_train)
#model.score(X, y_train)
#Predict Output
predictions= model.predict(X_test)
print(classification_report(y_test,predictions))
print(model.score(X_test,y_test))




#################with reduction ########################################
model = svm.SVC(decision_function_shape='ovo',max_iter=9000) 


###############linear svm model with pca ##########################################

"""
model=svm.LinearSVC(max_iter=90000)
"""
#########################################################################

model.fit(X_train_pca, y_train)
#model.score(X, y_train)
#Predict Output
print('pca.................................................................')
predictions= model.predict(X_test_pca)
print(classification_report(y_test,predictions))
print(model.score(X_test_pca,y_test))


"""
###########cross validation#############################################
clf = svm.SVC(decision_function_shape='ovo',max_iter=9000)

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores2= cross_val_score(clf, X, Y, cv=5)

#cross validation without custom parameters
#scores = cross_val_score(clf, X, Y, cv=10)
print(scores2)


"""
########################to save model##################################
"""
x=input()
if (x==1):
	joblib.dump(model,'svm-model-ovr.sav')
	print('saved')

"""
######################################################################