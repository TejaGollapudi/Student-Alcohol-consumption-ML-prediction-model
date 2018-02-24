import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import svm, datasets
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('mergedng-yn2binary-dummycoding.csv')
#print(df.describe().transpose())
#print(df.head())



X=df.drop('Walc',axis=1)
X=df.drop('Dalc',axis=1)
Y=df['Walc']
#print(X.describe().transpose())
#print(Y.describe().transpose())


X_train, X_test, y_train, y_test = train_test_split(X, Y)

#print(X_train)

#print(X_train)

model = svm.LinearSVC(max_iter=100000) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train, y_train)
#model.score(X, y_train)
#Predict Output
predictions= model.predict(X_test)


print(classification_report(y_test,predictions))
