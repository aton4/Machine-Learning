import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt('X_train.txt', delimiter=None)
Y = np.genfromtxt('Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)
Y = np.array(Y)

X_training, X_valid, Y_training, Y_valid = train_test_split(X, Y, test_size = 0.15)

estimators = [('rf', RandomForestClassifier(n_estimators=500, max_depth=20,n_jobs = 3,verbose=1)), 
              ('knn', KNeighborsClassifier(n_neighbors = 5, n_jobs = 3)),
              ('gb', HistGradientBoostingClassifier(max_iter=2200,early_stopping=False,verbose=1))]
lowest_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(),verbose=1).fit(X_training, Y_training)

TRpredict = lowest_clf.predict(X_training)
TR_MSE = np.mean(abs(TRpredict - Y_training))
print(f'Train - MSE', TR_MSE)

TEpredict = lowest_clf.predict(X_valid)
VA_MSE = np.mean(abs(TEpredict - Y_valid))
print(f'TEST - MSE for estimator', VA_MSE)

predict_probs = lowest_clf.predict_proba(X_valid)[:,1]
fp, tp, thresh = roc_curve(Y_valid, predict_probs)
plt.plot(fp, tp)
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC of validation data')
plt.show()

auc_score = roc_auc_score(Y_valid, predict_probs)
print("auc score: ", auc_score)

Xte = np.genfromtxt('X_test.txt', delimiter=None)
Yte = np.vstack((np.arange(Xte.shape[0]), lowest_clf.predict_proba(Xte)[:,1])).T
np.savetxt('Y_submit.txt',Yte,'%d,%.2f',header='ID,Prob1',comments='',delimiter=',')