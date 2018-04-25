import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from models.cnn import ConvNet

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from metrics.get_metrics import get_metrics, plot_acc_loss, plot_confusion_matrix, onehot2label

from sklearn import metrics
from sklearn.metrics import roc_auc_score
import gensim

#word2vec_path = "./embeddings/GoogleNews-vectors-negative300.bin.gz"
#word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

print("word2vec loaded")

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

cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(data["class"]))

indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data = cnn_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

#embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
#for word,index in word_index.items():
#    embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
#print(embedding_weights.shape)



#embedding_weights = np.load('./data/embeddings_original_weights.npy')
embedding_weights = np.load('./data/embeddings_weights.npy')

print("embeddings done")

#x_train = cnn_data[:-num_validation_samples]
#y_train = labels[:-num_validation_samples]
#x_val = cnn_data[-num_validation_samples:]
#y_val = labels[-num_validation_samples:]

x_train = np.load('./data/X_train_additional.npy')
y_train = np.load('./data/Y_train_additional.npy')
x_val = np.load('./data/X_val_additional.npy')
y_val = np.load('./data/Y_val_additional.npy')

model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM, 
                2)


history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=32)


predictions = model.predict(x_val)
preds = onehot2label(predictions)
labels= onehot2label(y_val)

cnf_matrix = confusion_matrix(labels, preds)

plot_confusion_matrix(cnf_matrix, classes=['not_flagged', 'flagged'],
                      title='Confusion matrix, without normalization')

plt.savefig("cnn_additional_cmatrix.png")

plot_acc_loss("original validation", [history.history], 'val_acc', 'val_loss' )

plt.savefig("additional_history.png")

print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(labels, preds)))
print("ROC: %s" % roc_auc_score(labels, preds) )