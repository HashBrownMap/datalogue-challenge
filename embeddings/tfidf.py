import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from plot_LSA import plot_LSA


def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

if __name__ == '__main__':
    
    
    data = pd.read_pickle('../data/processeddata.pickle')

    X_data = data['response_text'].tolist()
    y_data = data['class'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.15, random_state=40)

    X_train_counts, count_vectorizer = cv(X_train)
    X_test_counts = count_vectorizer.transform(X_test)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_test_tf = tf_transformer.transform(X_test_counts)

    fig = plt.figure(figsize=(12,12))
    plot_LSA(X_train_tf, y_train)
    plt.savefig("tfidf.png")

    sparse.save_npz('../data/X_train_tf.npz', X_train_tf)
    np.save('../data/Y_train_tf.npy', y_train)
    sparse.save_npz('../data/X_test_tf.npz', X_test_tf)
    np.save('../data/Y_test_tf.npy', y_test)


