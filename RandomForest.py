import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


X = np.genfromtxt('X_train.txt', delimiter=None)
Y = np.genfromtxt('Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)
Y = np.array(Y)

X_training, X_valid, Y_training, Y_valid = train_test_split(X, Y, test_size = 0.15)

lowest_mse_va = 999999999
bestMaxDepth = -1
lowest_rf = RandomForestClassifier(n_estimators = 500, max_depth = -1, n_jobs = 3, verbose = 1)
mse_tr = []
mse_va = []
for depthNum in range(1, 31):
    print(f'Starting Random Forest With Depth {depthNum}')
    print(f'lowest rf depth: {lowest_rf.max_depth}')

    rf = RandomForestClassifier(n_estimators = 500, max_depth = depthNum, n_jobs = 3, verbose = 1)
    rf.fit(X_training, Y_training)

    # predictions on training data
    TRpredict = rf.predict(X_training)
    TR_MSE = np.mean(abs(TRpredict - Y_training))
    print(f'Train - MSE for depth {depthNum}:', TR_MSE)
    mse_tr.append(TR_MSE)

    # predictions on validation data
    TEpredict = rf.predict(X_valid)
    VA_MSE = np.mean(abs(TEpredict - Y_valid))
    print(f'TEST - MSE for depth {depthNum}:', VA_MSE)
    mse_va.append(VA_MSE)

    if (VA_MSE < lowest_mse_va):
        lowest_mse_va = VA_MSE
        bestMaxDepth = depthNum
        lowest_rf = rf

# best is with 500 estimators max depth = 20
print("best max depth: ", bestMaxDepth)
print("lowest rf max depth: ", lowest_rf.max_depth)
figObj, plotObj = plt.subplots(nrows = 1, ncols = 2)
plotObj[0].plot([num for num in range(1,31)], mse_tr, "red")
plotObj[0].plot([num for num in range(1,31)], mse_va, "green")
plotObj[0].set_title("MSE Plot")
plotObj[0].set_xlabel("maxDepth")
plotObj[0].set_ylabel("Mean Squared Error")

plotObj[1].semilogx([num for num in range(1,31)], mse_tr, "red")
plotObj[1].semilogx([num for num in range(1,31)], mse_va, "green")
plotObj[1].set_title("MSE semilogx")
plotObj[1].set_xlabel("maxDepth")
plotObj[1].set_ylabel("Mean Squared Error")
plt.show()

predict_probs = lowest_rf.predict_proba(X_valid)[:,1]
fp, tp, thresh = roc_curve(Y_valid, predict_probs)
plt.plot(fp, tp)
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC of validation data')
plt.show()    

auc_score = roc_auc_score(Y_valid, predict_probs)
print("auc score: ", auc_score)

print("lowest rf max depth: ", lowest_rf.max_depth)
Xte = np.genfromtxt('X_test.txt', delimiter=None)
Yte = np.vstack((np.arange(Xte.shape[0]), lowest_rf.predict_proba(Xte)[:,1])).T
np.savetxt('Y_submit.txt',Yte,'%d,%.2f',header='ID,Prob1',comments='',delimiter=',')
