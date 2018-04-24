import numpy as np
from scipy import sparse
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

X_train = np.load('../data/X_train_w2v.npy')
y_train = np.load('../data/Y_train_w2v.npy')
X_test = np.load('../data/X_test_w2v.npy')
y_test = np.load('../data/Y_test_w2v.npy')
X_train, y_train = shuffle(X_train, y_train)

svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)

svm_predict_valid = np.array(svm_clf.predict(X_train))
svm_predict_test = np.array(svm_clf.predict(X_test))

print("F1-measure for validation using Linear SVM classifier: %f" % f1_score(svm_predict_valid, y_train, average="micro"))
print("F1-measure for test using Linear SVM classifier: %f" % f1_score(svm_predict_test, y_test, average="micro"))
