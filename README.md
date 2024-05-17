# Named Entity Recognition
## AIM
To develop an LSTM-based model for recognizing the named entities in the text.
## Problem Statement and Dataset
Our aim in this experiment is to develop an advanced model employing Bidirectional Recurrent Neural Networks (RNNs) within a Long Short-Term Memory (LSTM) framework. We aim to train this model to accurately detect named entities within textual data. Our dataset consists of numerous sentences, each containing multiple words with associated tags. Leveraging embedding techniques, we convert these words into vectors, facilitating the training process. Bidirectional RNNs enable us to connect two hidden layers in opposing directions to the same output. This architecture allows the output layer to incorporate information from both preceding and succeeding states simultaneously, thereby enhancing the model's predictive capabilities.
![image](https://github.com/Visalan-H/named-entity-recognition/assets/152077751/1b824a8c-57d1-45d8-bef0-51f15e1620f2)
## DESIGN STEPS
### STEP 1:
Import the necessary packages.
### STEP 2:
Read the dataset, and fill the null values using forward fill.
### STEP 3:
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.
### STEP 4:
Create a dictionary for the words and their Index values. Do the same for the tags as well,Nowwe move to moulding the data for training and testing.
### STEP 5:
We do this by padding the sequences,This is done to acheive the same length of input data
### STEP 6:
We build a build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.
### STEP 7:
We compile the model and fit the train sets and validation sets,We plot the necessary graphs for analysis,A custom prediction is done to test the model manually.
## PROGRAM
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data=pd.read_csv("ner_dataset.csv",encoding="latin1")
data.head(50)
data=data.fillna(method="ffill")
data.head(50)

words=list(data["Word"].unique())
words.append("ENDPAD")
tags=list(data["Tag"].unique())
num_words=len(words)
num_tag=len(tags)

class SentenceGetter(object):
  def __init__(self,data):
    self.n_sent=1
    self.data=data
    self.empty=False
    agg_func=lambda s:[(w,p,t) for w, p, t in zip(s["Word"].values.tolist(),
                                                  s["POS"].values.tolist(),
                                                  s["Tag"].values.tolist())]
    self.grouped=self.data.groupby("Sentence #").apply(agg_func)
    self.sentences=[s for s in self.grouped]
  def get_next(self):
    try:
      s=self.grouped["Senatence: {}".format(self.n_sent)]
      self.n_sent+=1
      return s
    except:
      return None

getter=SentenceGetter(data)
sentences=getter.sentences
len(sentences)
word2Idx={w: i+1 for i,w in enumerate(words)}
tag2Idx={t: i for i,t in enumerate(tags)}
plt.hist([len(s) for s in sentences], bins=50)
plt.show()
x1=[[word2Idx[w[0]] for w in s] for s in sentences]
max_len=50
X=sequence.pad_sequences(maxlen=max_len,
                         sequences=x1,padding="post",
                         value=num_words-1)
y1=[[tag2Idx[w[2]] for w in s]for s in sentences]
y=sequence.pad_sequences(maxlen=max_len,
                          sequences=y1,
                          padding="post",
                          value=tag2Idx["O"])
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)
x_train[0]
y_train[0]
input_word=layers.Input(shape=(max_len,))
embedding_layer=layers.Embedding(input_dim=num_words,
                                 output_dim=50,
                                 input_length=max_len
)(input_word)
dropout_layer=layers.SpatialDropout1D(0.2)(embedding_layer)
bidirectional_lstm=layers.Bidirectional(
    layers.LSTM(units=150,
                return_sequences=True,
                recurrent_dropout=0.3)
)(dropout_layer)
output=layers.TimeDistributed(
    layers.Dense(num_tag,activation="softmax")
)(bidirectional_lstm)
model=Model(input_word,output)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history=model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test,y_test),
    batch_size=32,
    epochs=3,
)
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
print("Name: Visalan H")
print("Ref no.: 212223240183")
i = 20
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))
```
## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Visalan-H/named-entity-recognition/assets/152077751/d3f78028-8ac4-4db1-9f8f-c70d4dbc4f91)
![image](https://github.com/Visalan-H/named-entity-recognition/assets/152077751/41ea7827-c5f1-48d9-8f49-1cb65364c24d)
### Sample Text Prediction
![image](https://github.com/Visalan-H/named-entity-recognition/assets/152077751/ff3148c8-a3c2-4cbd-ae5d-82e4bcf3becf)
## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text is successfully developed.
