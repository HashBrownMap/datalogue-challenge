
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from models.cnn import ConvNet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


data = pd.read_pickle('./data/newprocesseddata.pickle')

all_words = [word for tokens in data["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in data["tokens"]]
VOCAB = sorted(list(set(all_words)))

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 35
VOCAB_SIZE = len(VOCAB)

VALIDATION_SPLIT=.2
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(data["response_text"].tolist())
sequences = tokenizer.texts_to_sequences(data["response_text"].tolist())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

embedding_weights = np.load('./data/embeddings_weights.npy')

x_train = np.load('./data/X_train_additional.npy')
y_train = np.load('./data/Y_train_additional.npy')
x_val = np.load('./data/X_val_additional.npy')
y_val = np.load('./data/Y_val_additional.npy')

print("loaded embeddings")

learn_rate=[0.001, 0.01, 0.05]
epochs=[10,20,30]
batch_size=[16,32.64]
dropout = [0.3, 0.4, 0.5]

cnn = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM)

model = KerasClassifier(build_fn=cnn)

param_grid = dict(epochs=epochs, learn_rate=learn_rate, dropout = dropout, batch_size=batch_size)

#model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train, validation_data=(x_val, y_val))


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))