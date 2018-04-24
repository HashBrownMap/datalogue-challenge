import gensim
import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.model_selection import train_test_split
from plot_LSA import plot_LSA

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, data, generate_missing=False):
    embeddings = data['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


if __name__ == '__main__':
    
    word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    data = pd.read_pickle('../data/processeddata.pickle')
    y_data = data['class'].tolist()

    embeddings = get_word2vec_embeddings(word2vec, data) 
    
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y_data, test_size=0.15, random_state=40)

    fig = plt.figure(figsize=(12, 12))          
    plot_LSA(X_train, y_train)
    plt.savefig("original_w2v.png")

    np.save('../data/X_train_original_w2v.npy', X_train)
    np.save('../data/Y_train_original_w2v.npy', y_train)
    np.save('../data/X_test_original_w2v.npy', X_test)
    np.save('../data/Y_test_original_w2v.npy', y_test)