Sentiment Analysis
=====================

A neural network to classify if a chatbot should continue to chat or refer for help. 

How To Run
------------

1. Install dependencies  
```pip install -r requirements.txt```
2. To execute, run   
    ```python play.py```.

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

 
We decided to use a Convolutional Neural Network. Reason we didn't use an RNN was due to the constraint of time and size of the dataset. Through the [paper](https://arxiv.org/pdf/1408.5882.pdf) by Kim, Yoon, it is shown that a CNN can perform fairly well with text classification problems and can be trained much faster compared to an rnn. We will tune learning rate, dropout and kernel size.  
For word representation, we will try TFIDF and Word Embeddings [pre-trained embedding](https://code.google.com/archive/p/word2vec/). 

Results
------------

The results below all used Word2Vec embeddings.    

 Models | Validation Accuracy 
 -------- | ---------- 
 Logistic Regression | 0.750
 Linear SVM | 0.750 |
 CNN with original dataset | 0.875

 Models on extended dataset | Validation Accuracy | ROC | precision | recall
 -------- | ---------- | --------- | ------- | --------
 Logistic Regression| 0.944 | 0.943 | 0.940 | 0.940
 Linear SVM | 0.930 | 0.959 | 0.960 | 0.960 
 CNN | 0.986 | 0.957 | 0.960 | 0.960

With the original data, the baseline models performed poorly. Outliers made it hard for them to perform well. As to the word representation approach, we can see that the word embeddings is a better approach through the following graphs.  


 TFIDF             |  Word2Vec Embedding(Non-Static)     
:-------------------------:|:-------------------------: 
![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/embeddings/tfidf.png)  |  ![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/embeddings/w2v.png) 

 
 After running a truncated SVD, we see that the words weight of each class are still clustered together in TFIDF compared to a Word2Vector embedding. The CNN also performed badly under the original data. It would overfit on the training data and perform badly on the validation data.    


![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/original_history.png)

After data augmentation, validation would increase but it's actually misleading us.   

![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/history.png)

It's misleading due to our approach in generating more data. For the small dataset we had, we identified words with more weight in each class and used their synonyms to create new data. However there were a few samples which didn't contain those words. Thus no new samples were generated using those respective samples. So our model would overfit on the samples that were used to generate data and newly generated data. Our validation set would also include mostly new data. This resulted in such high validation accuracy.  

Logistic Regression             |  Linear SVM           | CNN
:-------------------------:|:-------------------------: | :------------------:
![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/models/lr_additional_cmatrix.png)  |  ![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/models/svm_additional_cmatrix.png) | ![oops figure not showing](https://github.com/HashBrownMap/datalogue-challenge/blob/master/cnn_additional_cmatrix.png)

It is too note that Linear SVM had the least false negative(Type II). This is the most crucial in our situation as we want to seek help immediately if the messaged is flagged. However it is to note that the validation dataset was far too small to be able to make the conclusion that the SVM model is a better performer.

Future Work
------------

Given more time, here are some possible expansion:
* Perhaps crawl Twitter and get more data related to depression.
* With more data and time, implement a Recurrent Neural Network model. 
* cross-validate hyperparameters
* Perhaps improve the accuracy of pre-trained word embeddings for this [paper](https://arxiv.org/pdf/1711.08609.pdf). Popular word embeddings methods such as Word2Vec and Glove currently ignore sentiment information of texts. The paper suggests a new method, Improved Word Vectors, which claims to be more effective for sentiment analysis.
