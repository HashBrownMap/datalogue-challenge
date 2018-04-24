Sentiment Analysis
=====================

A neural network to classify if a chatbot should continue to chat or refer for help. 

How To Run
------------

To execute, run `python play.py`.

Method
------------

Data Processing
------------

To clean up the data we did the following:
* removed punctuation
* lowercased every word
* removed stopped words
* lemmatized words to their corresponding base words  

With the size of the given dataset (80), a neural network model wouldn't perform so well. Thus we had to find a way to generate new data. To do so, we first identified key words for each class. Then for each key word we got their synonyms using Spacy. We then created new sentences by replacing key words by their synonyms. We were able to expand our dataset to 360 samples.  

the steps above were done in `preprocessing.ipynb`  

For word representation, we tried:
* Term Frequency Inverse Document Frequency
* Word2Vector embeddings

Models
------------

As baseline models we tried:
* Logistic Regression
* Linear SVM  

 
We decided to use a Convolutional Neural Network. Reason we didn't use an RNN was due to the constraint of time. Through the (paper)[https://arxiv.org/pdf/1408.5882.pdf] by Kim, Yoon, it is shown that a CNN can perform fairly well with text classification problems. We will tune learning rate, dropout and kernel size. We will use a (pre-trained embedding)[https://code.google.com/archive/p/word2vec/]. 

Results
------------

With the original data, the baseline models performed poorly. After running a truncated SVD, we see that the words weight of each class are still cluster together in TFIDF compared to a Word2Vector embedding. The CNN also performed badly under the original data. It would overfit on the training data and perform badly on the validation data.    

![alt text](https://github.com/HashBrownMap/datalogue-challenge/blob/master/original_history.png)

After data augmentation, validation would increase but it's actually misleading us.   

![alt text](https://github.com/HashBrownMap/datalogue-challenge/blob/master/history.png)

It's misleading due to our approach in generating more data. For the small dataset we had, we identified words with more weight in each class and used their synonyms to create new data. However there were a few samples which didn't contain those words. Thus no new samples were generated using those respective samples. So our model would overfit on the samples that were used to generate data and newly generated data. Our validation set would also include mostly new data. This resulted in such high validation accuracy.  

The results below all used Word2Vec embeddings.    

| Models | Accuracy | 
| -------- | :---------- :|
| Logistic Regression | 0.750 |
| Linear SVM | 0.750 |
| CNN with original dataset | 0.875 |
| CNN with extended dataset | 0.986 |


Future Work
------------

Future Work:
Given more time, here are some possible expansion:
* Perhaps crawl Twitter and get more data related to depression.
* Generate confusion matrix
* With more data, implement a Recurrent Neural Network model. 
