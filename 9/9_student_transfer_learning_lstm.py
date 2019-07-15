# -*- coding: utf-8 -*-
"""9_STUDENT_transfer_learning_lstm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SpvbjN-yHY-b-zfnp7Yfw7Hs6ZJZArv8

# Tutorial 9: Recurrent Neural Networks (RNNs) and Transfer Learning

Welcome to the ninth tutorial of the course 'Machine learning for Precision Medicine'.

In this exercise we will touch on two important topics in modern machine learning research: RNNs and transfer learning.

We have prepared a dataset for you, consisting of 4104 drug-reviews with corresponding (subjective) ratings and effectiveness scores. This dataset was derived from the **Drug Review Dataset (Druglib.com)** on the [UCI Machine Learning Repository ](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+(Druglib.com)).

Your task will be to build a predictor that is able to predict the rating from 1 to 10. You will be free to implement a second model that predicts the effectiveness.

To build this predictor, we will first encode the reviews using sentence embedding. Every sentence of each review will be encoded as a vector of 512 numbers, using a pre-trained *universal sentence encoder* available on [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder/2) and published [arXiv](https://arxiv.org/abs/1803.11175). These vectors represent the content of each sentence in a dense vector. The universal sentence encoder was trained on a large corpus of text - much larger than our training set.

The model that encodes our sentences was trained with the Stanford Natural Language Inference (SNLI) corpus. The SNLI corpus is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE). Essentially, the model was trained to learn the semantic similarity between the sentence pairs.

After embedding the sentences, aach review (which will then be represented as a sequence of embeddings), will be fed into an LSTM in order to predict the rating.

The process of taking a model that was trained on a different task and using it for a new task is called transfer learning. Often the pre-trained model was trained with a lot more data than is availble for the new task. Transfer learning is usually a good idea if the number of training examples is small.
"""

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from keras.layers import Input, Lambda, Dense, LSTM, RNN, GRU, Dropout
from keras import Model
from keras.models import Sequential
import keras.backend as K
from keras.optimizers import Adam

# Import the Universal Sentence Encoder's TF Hub module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)

# we load the data and display a couple of reviews
drugreviews = pd.read_csv('https://raw.githubusercontent.com/remomomo/mlpm/master/data_proc.tsv', sep='\t', header=0)

for i in [0, 10, 1500, 3000]:
  print('\ndrug review example:')
  print("{}".format(drugreviews['benefitsreview'][i]))
  print('# rating: {}, effectiveness: {}'.format(drugreviews['rating'][i], drugreviews['effectiveness'][i]))

"""As we can see above, the reviews cover different types of drugs and have different lengths. Estimating the rating from the review is not a straight forward task and probably can't be solved perfectly. How well could you perform this? In the following cells, we will perform some additional pre-processing steps in order to split the reviews in to single sentences."""

# we perform some additional pre-processing
reviews_split = [review.replace('i.e.','ie').replace('...','.').split('.') for review in drugreviews['benefitsreview']]
reviews_split = [ [sentence for sentence in review if len(sentence) > 8 ] for review in reviews_split ]
print("review:\n {}".format(reviews_split[3000]))
print('# length: {}'.format(len(reviews_split[3000])))

# we limit ourselves to the first 10 sentences of every review:
reviews_split = [ review[:min(10, len(review))] for review in reviews_split ]

_ = plt.hist(np.array(list(map(len, reviews_split))), 10)
_ = plt.title('length of drug reviews')

"""The following cell wraps the embed-function imported above in a Keras model. We will pre-compute the embeddings for all observations using this model. Additionally, we will zero-pad the sequences to all have length (10). This will later facilitate the training process."""

def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
    	signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
out = Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
sequence_embedder = Model(input_text, out)

all_sentences = np.concatenate(reviews_split)
print('shape before embedding and padding: {}'.format(all_sentences.shape))

# we pre-compute the embeddings for all the sentences:
with tf.Session() as session:

  K.set_session(session)
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  
  embeddings = sequence_embedder.predict(all_sentences, verbose=1)

"""Now that we have computed the embeddings for all sentences, we can visualize a couple of examples."""

import seaborn as sns

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")

plot_similarity(all_sentences[[80,20,200,40,100]], embeddings[[80,20,200,40,100]], 90)

# reshaping back...
embedded_reviews = []
i = 0
for review in reviews_split:
  embedded_reviews.append(embeddings[i:(i+len(review))])
  i+=len(review)
embedded_reviews = np.array(embedded_reviews)

for i in [0, 10, 1500, 3000]:
  print('\ndrug review example:')
  print("{}".format(drugreviews['benefitsreview'][i]))
  print('# rating: {}, effectiveness: {}'.format(drugreviews['rating'][i], drugreviews['effectiveness'][i]))
  print('# embedding shape: {}'.format(embedded_reviews[i].shape))

embedded_reviews_pad = np.array([np.pad(embedded_review, ((10-embedded_review.shape[0], 0), (0,0)), mode='constant', constant_values=0.) for embedded_review in embedded_reviews ])
print('shape after embedding and padding: {}'.format(embedded_reviews_pad.shape))

"""The data has now been pre-processed and is ready to be used to train our model - a recurrant neural network:

## LSTM
RNNs are usually designed to learn patterns in sequential data, i.e. text, genomes, sensor data, ...
LSTMs (Long-shortterm memory) models are a type of RNNs to model sequential data. Sequential data follows some structure, i.e. language follows grammar rules. With a CNN only the current sample (i.e. image) is used as an input. RNN's however, also take previously seen samples as input for the feed-forward input, which adds together as two input sources, present and past data with the aim to predict new data. To feed back past data to the input, the the output of a neural network layer at time t goes back to the input of the same network layer at time t + 1. This is what is considered as memory. The sequential information is preserved in the recurrent network’s hidden state and essentially what RNNs do is sharing weights over time.

Here we will learn the sequential structure in drug reviews. 

In the lecture you heard about the different components of an LSTM, namely input gate, the forget gate and the output gate.

__Input gate:__  
First, we will squash this combined input through a tanh-activated layer resulting in values between -1 and 1. This can be expressed by:

$$ g=tanh(b^g + x_tU^g+h_{t−1}V^g),$$


where $U^g$ and $V^g$ are the weights for the input and previous cell output, respectively, and $b^g$ is the input bias. Note that the exponents g are not a raised power, but rather signify that these are the input weights and bias values.

We pass this $g$ through the input gate, which is basically a layer of sigmoid activated nodes. The idea behind these sigmoid activated nodes is to exclude elements of the input vector that are not relevant (pass the associated weights when sigmoid output of this node is close to 1, and exclude when close to 0). Then we do an element-wise multiplication of the output and the squashed input.

$$i = \sigma(b^i+x_tU^i+h_{t-i}V^i) $$  and  
$$g\cdot i$$


__Forget gate:__  
In the next step, we add the internal state variable st, which lagged one time step i.e. st−1, to the input data and pass it through the forget gate. This is an addition operation (not multiplication) reduces the risk of vanishing gradients. In the forget layer the model learns which state variables should be “remembered” or “forgotten”.

The output of the forget gate is expressed as:
$$f=\sigma(b^f+x_tU^f+h_{t-1}U^f)$$

The output of the element-wise product of the previous state and the forget gate is expressed as st−1∘f. The output from the forget gate / state loop stage is
$$s_t=s_{t-1}\cdot f+g\cdot i $$


__Output gate:__  
Finally, we arrive at the output gate, which determines which values are actually allowed as an output from the cell ht. 

The output gate operation is expressed as   

$$o =\sigma(b^o+x_tU^o+h_{t-1}V^o) $$

and the final output results by

$$h_t = tanh(s_t)\cdot o$$

Luckily, you do not have to implement these different gates yourself. Keras provides a layer `LSTM`, which can have multiple hidden units connected by the gates described above.

## Task 1:

Complete the function `get_LSTM()` below, that builds a model using a single LSTM layer with 4 hidden units. The ouput of the LSTM layer, is passed to a dense layer in order to predict the rating from 1 to 10. Choose the appropriate size for the dense layer, and specify the correct activation.

You should do this using the keras functional API. Read up on how to define a keras `Model` with the [functional API](https://keras.io/models/model/).
"""

def get_LSTM():
    
    in_layer = Input(shape=(10,512))
    
    # your code  Input, Lambda, Dense, LSTM, RNN, GRU
    lstm_layer = LSTM(4)(in_layer)
    out_layer = Dense(1, activation="relu")(lstm_layer)
    model = Model(inputs=in_layer, outputs=out_layer)
    
    return model

"""We split the data in to train- and test sets:"""

ratings = np.array(drugreviews['rating'])

train_reviews = embedded_reviews_pad[0:3500]
train_ratings = ratings[0:3500]

test_reviews = embedded_reviews_pad[3500:]
test_ratings = ratings[3500:]

"""## Task 2:
Complete the cell below. First retrieve your model with the function you defined above. Compile the model using the Adam optimizer and a learning rate of 0.01. Fit the model for 30 epochs, use the mean squared error as your loss and and monitor the performance on the validation data (`test_reviews`, `test_ratings`) using the mean absolute error during training.
"""

with tf.Session() as session:
  
  K.set_session(session) 
  lstm_model = get_LSTM()# your code (load the model)
  
  # your code (compile and fit)
  optimizer = Adam(lr=0.01)
  lstm_model.compile(optimizer, loss="mean_squared_error", metrics=['accuracy', 'mean_absolute_error'])
  lstm_model.fit(x=train_reviews, y=train_ratings, batch_size=15, epochs=30, validation_split=0.15)
  
  preds = lstm_model.predict(test_reviews)

"""**Expected Output (approximately):**

```
Train on 3500 samples, validate on 604 samples
Epoch 1/30
3500/3500 [==============================] - 3s 991us/step - loss: 17.7917 - mean_absolute_error: 3.6456 - val_loss: 8.9291 - val_mean_absolute_error: 2.6259
Epoch 2/30
3500/3500 [==============================] - 1s 308us/step - loss: 8.3288 - mean_absolute_error: 2.4174 - val_loss: 8.4183 - val_mean_absolute_error: 2.4193
Epoch 3/30
3500/3500 [==============================] - 1s 313us/step - loss: 7.7538 - mean_absolute_error: 2.3174 - val_loss: 7.3692 - val_mean_absolute_error: 2.2865

........

Epoch 28/30
3500/3500 [==============================] - 1s 295us/step - loss: 4.1380 - mean_absolute_error: 1.5699 - val_loss: 6.8988 - val_mean_absolute_error: 2.0348
Epoch 29/30
3500/3500 [==============================] - 1s 300us/step - loss: 4.0851 - mean_absolute_error: 1.5492 - val_loss: 6.9087 - val_mean_absolute_error: 2.0201
Epoch 30/30
3500/3500 [==============================] - 1s 296us/step - loss: 4.0451 - mean_absolute_error: 1.5426 - val_loss: 6.8554 - val_mean_absolute_error: 2.0641

```

### Question 1:  

Based on the output above, do you think the model is over or under-fitting?

Give an example of a "base-line model" that you could compare against. The model would use the same input as your sequential model.

Write your anser in the cell below:

The base-line model could be any classifier, for example a support vector machine. Seeing the quite high loss, the model seems to be under-fitting.
"""

# we plot the predicted vs actual ratings on the validation set:
from matplotlib import pylab

x, y = test_ratings, preds[:,0]
pylab.plot(test_ratings,preds,'o')

# calc the trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

_ = pylab.plot(x,p(x),"r--")

"""As mentioned in the introduction, the DrugReviews dataset contains another variable 'effectiveness', in which patients rated the effectiveness of the drug they were given.

They had to rate the effectiveness on a scale from "ineffective" to "highly effective":
"""

drugreviews['effectiveness'].unique()

"""### Question 2:  

Name two ways in which you could encode the variable "effectiveness". Write them in the cell below.

one-hot encoding or assigning the numbers from 0-5.

## Task 3:

Now that you have gathered experience with Keras over the last tutorials, you will be given a Task to perform more freely than usual.

Train a model of your choice that predicts the effectiveness of a drug based on the review. This will require you to perform the following tasks:


1.   Numerically encode the variable "effectiveness" in a way that it can be used in a machine-learning setting
2.   Define a model architecture that takes the sequences of sentence embeddings computed above, and calculates the effectiveness. This may or may not be a sequential model. Try playing around!
3.   Define the appropriate loss function and train a model using the first 3500 observations in the dataset as the training set, and the remaining as the validation set (as we did above).

You can leave feedback and a small conclusion of this task in the Feedback field below. Was it easy or difficult? Where did you have problems?

In the collab notebook, it might be necessary to run your code as we did above:


```
with tf.Session() as session:
      K.set_session(session)
      # define the model
      # fit the model
      # evaluate the model
```
"""

drugreviews["effectiveness"] = drugreviews["effectiveness"].astype('category')
drugreviews["effectiveness_cat"] = drugreviews["effectiveness"].cat.codes
print(drugreviews.head())

train_effectiveness = np.array(drugreviews['effectiveness_cat'][:3500])
test_effectiveness = np.array(drugreviews['effectiveness_cat'][3500:])

with tf.Session() as session:
      K.set_session(session)
      # define the model

      in_layer = Input(shape=(10,512))
        
      lstm_layer = LSTM(10)(in_layer)
        
      a = Dense(100, activation="relu")(lstm_layer)
      a1 = Dropout(0.3)(a)
      b = Dense(50, activation="relu")(a1)
      out_layer = Dense(1, activation="relu")(b)

    
    # your code  Input, Lambda, Dense, LSTM, RNN, GRU
    
      model = Model(inputs=in_layer, outputs=out_layer)
      # fit the model
      optimizer = Adam(lr=0.01)
      model.compile(optimizer, loss="mean_squared_error", metrics=['accuracy', 'mean_absolute_error'])
      model.fit(x=train_reviews, y=train_effectiveness, batch_size=5, epochs=30, validation_split=0.15)
      # evaluate the model

"""## Feedback:

For **Task 3** write your conclusions and feedback in the cell below
"""



"""# Submitting your assignment

Please save your notebook under your full name and **submit it on the moodle platform**.

Please rename the file to 9_transfer_learning_lstm_<GROUP\>.ipynb and replace <GROUP\> with your group-name. (File -> Download .ipynb)

As this is also the first time for us preparing this tutorial, you are welcome to give us feedback to help us improve this tutorial.  

Thank you!  

Jana & Remo
"""