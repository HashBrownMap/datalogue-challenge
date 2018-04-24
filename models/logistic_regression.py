import numpy as np
from scipy import sparse
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


X_train = np.load('../data/X_train_w2v.npy')
y_train = np.load('../data/Y_train_w2v.npy')
X_test = np.load('../data/X_test_w2v.npy')
y_test = np.load('../data/Y_test_w2v.npy')
X_train, y_train = shuffle(X_train, y_train)


logistic_reg_clf = LogisticRegression()
logistic_reg_clf.fit(X_train, y_train)

train_preds = logistic_reg_clf.predict(X_train)
test_preds = logistic_reg_clf.predict(X_test)

#accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(y_test, test_preds)

print("F1-measure for validation using Logistic classifier: %f" % f1_score(train_preds, y_train, average="micro"))
print("F1-measure for test using Logistic classifier: %f" % f1_score(test_preds, y_test, average="micro"))
#print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, 
#                                                                       recall_word2vec, f1_word2vec))