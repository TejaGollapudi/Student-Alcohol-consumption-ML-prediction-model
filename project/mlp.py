import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA ,KernelPCA
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import RepeatedKFold
import scikitplot as skplt
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score





df=pd.read_csv('mergedng-yn2binary-dummycoding.csv')
#print(df.describe().transpose())
#print(df.head())

X=df.drop('Walc',axis=1)

X=df.drop('Dalc',axis=1)

red_time=[]
red_acc=[]
red_a=[]
a=[]
tim=[]
acc=[]
ind=[1,2,3,4,5,6,7,8,9,10]


################################LEARNING CURVE#####################
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	plt.figure()
	plt.title(title)
	#if ylim is not None:
		#plt.ylim(*ylim)
	plt.xlabel('Training examples')
	plt.ylabel('Score')
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()
	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color='r')
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color='g')
	plt.plot(train_sizes, train_scores_mean, 'o-', color='r',label='Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color='g',label='Cross-validation score')
	plt.legend(loc='best')
	return plt
#################################################################################
def avg(l):
	a=reduce(lambda x, y: x + y, l) / len(l)
	return format(a, '.3f')
###################################################
X=df.drop('school_GP',axis=1)
X=df.drop('school_MS',axis=1)
scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_t = scaler.transform(X)

pca2=PCA(.99)
pca2.fit(X_t)
X_pca=pca2.transform(X_t)

Y=df['Dalc']
#print(X.describe().transpose())
#print(Y.describe().transpose())

for i in range(5):
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

	################PCA########################################
	#pca = PCA(.99)
	pca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
	pca.fit(X_train)
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	print(X_test_pca.shape)


	#################### MLP without reduction###############



	mlp = MLPClassifier(hidden_layer_sizes=(40),max_iter=2000,solver='lbfgs',tol=1e-10,activation='tanh',alpha=0.01)
	start = time.time()
	mlp.fit(X_train,y_train)
	end = time.time()
	print('time consumed')
	print(end - start)
	tim.append(end-start)


	predictions = mlp.predict(X_test)
	print(' without dimensionality reduction ***************************************************************')
	print(classification_report(y_test,predictions))
	#print(mlp.score(X_test,y_test))
	acc.append(accuracy_score(y_test,predictions))
	a.append(f1_score(y_test,predictions,average='macro'))
	fpr2 = dict()
	tpr2 = dict()
	roc_auc2 = dict()
	predictionsd2=pd.get_dummies(predictions)
	y_testd2=pd.get_dummies(y_test)

	################### MLP with reduction####################
	mlp = MLPClassifier(hidden_layer_sizes=(21),max_iter=20000,solver='lbfgs',tol=1e-10,activation='tanh',alpha=0.01)
	start = time.time()
	mlp.fit(X_train_pca,y_train)
	end=time.time()
	print('time consumed')
	print(end-start)
	predictions = mlp.predict(X_test_pca)
	print(' with dimensionality reduction ***************************************************************')
	print(classification_report(y_test,predictions))
	#print(mlp.f_score(X_test_pca,y_test))
	print(f1_score(y_test,predictions,average='macro'))
	#y_score=mlp.predict(X_test_pca)
	red_time.append(end-start)
	red_acc.append(accuracy_score(y_test,predictions))
	red_a.append(f1_score(y_test,predictions,average='macro'))
	
	
	print(predictions.shape)
	print(y_test.shape)
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	predictionsd=pd.get_dummies(predictions)
	y_testd=pd.get_dummies(y_test)
	#print(predictionsd.shape)
	#print(y_testd.shape)
	#predictionsd=predictionsd.values
	#y_testd=y_testd.values
	#print(predictionsd.shape)
	#print(y_testd.shape)
	
	

#########ROC ASSIGNMENT################
for j in range(5):
    	fpr[j], tpr[j], abc = roc_curve(y_testd.iloc[:, j], predictionsd.iloc[:, j])
    	roc_auc[j] = auc(fpr[j], tpr[j])
for j in range(5):
    	fpr2[j], tpr2[j], abc2 = roc_curve(y_testd2.iloc[:, j], predictionsd2.iloc[:, j])
    	roc_auc2[j] = auc(fpr2[j], tpr2[j])
############TIME VS FSCORE PLOT#################################



plt.plot(red_a,red_time,'o-',color='g',label='model with dimensionality reduction ('+avg(red_a)+','+avg(red_time)+')')
plt.plot(a,tim,'o-',color='r',label='model without dimensionality reduction ('+avg(a)+','+avg(tim)+')')
plt.title('Fscore vs time taken to train')
plt.ylabel('time taken to train the nueral network in seconds')
plt.xlabel('F score ')
plt.legend(loc='best')
plt.show()



plt.plot(red_acc,red_time,'o-',color='g',label='model with dimensionality reduction ('+avg(red_acc)+','+avg(red_time)+')')
plt.plot(acc,tim,'o-',color='r',label='model without dimensionality reduction ('+avg(acc)+','+avg(tim)+')')
plt.title('Accuracy vs time taken to train, mean')
plt.ylabel('time taken to train the nueral network in seconds')
plt.xlabel('Accuracy')
plt.legend(loc='best')
plt.show()


############ROC PLOT#####################################
'''
for i in range(5):
    plt.figure()
    plt.plot(fpr[i], tpr[i],color='b',linewidth=7.0, label='ROC curve dimensionality reduction (area = %0.2f)' % roc_auc[i])
    plt.plot(fpr2[i], tpr2[i],color='r', label='ROC curve (area = %0.2f) ' % roc_auc2[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for class'+str(i+1))
    plt.legend(loc="lower right")
    plt.show()
'''
#################ACCURACY VS DATASPLIT########################
'''
plt.subplot(1, 2, 2)
plt.plot(ind,red_acc,'rs',label='model with dimensionality reduction')
plt.plot(ind,acc,'bs',label='model without dimensionality reduction')
plt.title('Accuracy vs the index of data split')
plt.ylabel('index of data split')
plt.xlabel('accuracy ')
plt.legend()
plt.show()
'''

####################cross validation######################

'''
clf = MLPClassifier(hidden_layer_sizes=(41),max_iter=2000,solver='lbfgs',tol=1e-15,activation='tanh')

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
scores2= cross_val_score(clf, X, Y, cv=5)

#cross validation without custom parameters##############
#scores = cross_val_score(clf, X, Y, cv=5)
print(scores2)
'''


#########################################################


#################to save model###########################

'''
x=input()
if (x==1):
	joblib.dump(mlp,'mlp-model.sav')
	print('saved')

'''

########################################################
#LCurve for REDUCED DATA#########################################################

'''
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = MLPClassifier(hidden_layer_sizes=(21),max_iter=2000,solver='lbfgs',tol=1e-10,activation='tanh',alpha=0.01)
plot_learning_curve(estimator, 'Learining curve after dimensionality reduction', X_pca, Y)

plt.show()

estimator2 = MLPClassifier(hidden_layer_sizes=(40),max_iter=2000,solver='lbfgs',tol=1e-10,activation='tanh',alpha=0.03)
plot_learning_curve(estimator2, 'Learining curve without dimensionality reduction', X, Y)

plt.show()
'''