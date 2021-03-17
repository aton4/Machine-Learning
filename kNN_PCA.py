import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

X = np.genfromtxt('X_train.txt', delimiter=None)
Y = np.genfromtxt('Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)
Y = np.array(Y)

X_training, X_valid, Y_training, Y_valid = train_test_split(X, Y, test_size = 0.15)

lowest_n_comp = -1
lowest_mse_va_ncomp = 999999999
lowest_knn = KNeighborsClassifier(n_neighbors = 1, n_jobs = 3)
lowest_k = -1
mse_va_ncomp = []
for n_comp in range(1, 14):
    pca = PCA(n_components = n_comp)
    pca.fit(X_training)
    X_pca_training = pca.transform(X_training)
    X_pca_valid = pca.transform(X_valid)

    mse_tr = []
    mse_va = []
    temp_lowest_mse_va_ncomp = 99999999
    for k in range(1, 37, 2):
        knn = KNeighborsClassifier(n_neighbors = k, n_jobs = 3)
        knn.fit(X_pca_training,Y_training)
        
        # predictions on training data
        TRpredict = knn.predict(X_pca_training)
        TR_MSE = np.mean(abs(TRpredict - Y_training))
        print(f'Train - MSE for k {k}:', TR_MSE)
        mse_tr.append(TR_MSE)

        # predictions on validation data
        TEpredict = knn.predict(X_pca_valid)
        VA_MSE = np.mean(abs(TEpredict - Y_valid))
        print(f'TEST - MSE for k {k}:', VA_MSE)
        mse_va.append(VA_MSE)

        if (VA_MSE < temp_lowest_mse_va_ncomp):
            temp_lowest_mse_va_ncomp = VA_MSE

        if (VA_MSE < lowest_mse_va_ncomp):
            lowest_mse_va_ncomp = VA_MSE
            lowest_n_comp = n_comp
            lowest_knn = knn
            lowest_k = k
    print(f'Valid - lowest MSE for ncomp {n_comp}:', temp_lowest_mse_va_ncomp)
    mse_va_ncomp.append(temp_lowest_mse_va_ncomp)
    print(f'Iterating n comp: {lowest_n_comp}, k: {lowest_k},{lowest_knn.n_neighbors}')

print(f'n comp: {lowest_n_comp}, k: {lowest_k},{lowest_knn.n_neighbors}')

figObj, plotObj = plt.subplots(nrows = 1, ncols = 2)
plotObj[0].plot([n_comp for n_comp in range(1, 14)], mse_va_ncomp, "green")
plotObj[0].set_title("MSE Plot")
plotObj[0].set_xlabel("n_comp")
plotObj[0].set_ylabel("Mean Squared Error")

plotObj[1].semilogx([n_comp for n_comp in range(1, 14)], mse_va_ncomp, "green")
plotObj[1].set_title("MSE semilogx")
plotObj[1].set_xlabel("n_comp")
plotObj[1].set_ylabel("Mean Squared Error")
plt.show()

# best is n_comp = 2, k = 5
pca = PCA(n_components = lowest_n_comp)
pca.fit(X_training)
X_pca_training = pca.transform(X_training)
X_pca_valid = pca.transform(X_valid)
lowest_knn = KNeighborsClassifier(n_neighbors = 5, n_jobs = 1)
lowest_knn.fit(X_pca_training,Y_training)

# predictions on training data
TRpredict = lowest_knn.predict(X_pca_training)
TR_MSE = np.mean(abs(TRpredict - Y_training))
print(f'Train - MSE for k {k}:', TR_MSE)

# predictions on validation data
TEpredict = lowest_knn.predict(X_pca_valid)
VA_MSE = np.mean(abs(TEpredict - Y_valid))
print(f'TEST - MSE for k {k}:', VA_MSE)

predict_probs = lowest_knn.predict_proba(X_pca_valid)[:,1]
fp, tp, thresh = roc_curve(Y_valid, predict_probs)
plt.plot(fp, tp)
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC of validation data')
plt.show()    

auc_score = roc_auc_score(Y_valid, predict_probs)
print("auc score: ", auc_score)

Xte = np.genfromtxt('X_test.txt', delimiter=None)
Xte = pca.transform(Xte)
print(len(lowest_knn.predict_proba(Xte)))
print(len(lowest_knn.predict_proba(Xte)[:,1]))
print(Xte.shape[0])
Yte = np.vstack((np.arange(Xte.shape[0]), lowest_knn.predict_proba(Xte)[:,1])).T
np.savetxt('Y_submit.txt',Yte,'%d,%.2f',header='ID,Prob1',comments='',delimiter=',')