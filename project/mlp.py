import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print(X_train)

mlp = MLPClassifier(hidden_layer_sizes=(41),max_iter=20000,solver='lbfgs',tol=1e-15,activation='tanh')
mlp.fit(X_train,y_train)



predictions = mlp.predict(X_test)
print(classification_report(y_test,predictions))
