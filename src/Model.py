from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import feature_column as fc
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Using data from the Titanic, our model aims to predict the likelihood of survival 
# We are using the training data to create the model and the testing data to evaluate our model

# A feature is an input variable used to train our model. In this case, our features are the characteristics of a passenger.
# A label is the output variable we are trying to predict. In this case, our label is whether or not a passenger survived.

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())   
print(dftrain.describe())
dftrain.age.hist(bins=20)     # plt.hist(dftrain['age'], bins=20)
plt.show()
dftrain.sex.value_counts().plot(kind='barh')    # same as dftrain['sex'].value_counts().plot(kind='barh')   
plt.show()                                      # barh means horizontal graph. 
dftrain['class'].value_counts().plot(kind='barh')
plt.show()

pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive') # Shows the likelihood that a person
# will survive based on their sex
plt.show()

"""
# Below code shows both graphs at once
# Create a figure and a subplot grid.
fig, axes = plt.subplots(nrows=1, ncols=2)
# Plot the age histogram on the first subplot.
dftrain.age.hist(bins=20, ax=axes[0])
# Plot the sex bar chart on the second subplot.
dftrain.sex.value_counts().plot(kind='barh', ax=axes[1])
# Tighten the layout.
plt.tight_layout()
# Show the figure.
plt.show()
"""

# Our data set has categorial data and numeric data. We want to convert the categorial data into numeric data so that our model can understand it
# We can do this by encoding each category with an integer (ex. male = 1, female = 2). Fortunately for us TensorFlow has some tools to help!
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck','embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()    # If feature_name == 'sex', vocabulary = ['male', 'female']
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) # Creates feature column for each categorical column

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32)) # Creates feature column for each categorical column

print(feature_columns)

# feed small batches of data to our model according to the number of epochs. An epoch is simply one stream of our entire dataset. 
# The number of epochs defined is the amount of times our model will see the entire dataset. We use multiple epochs in hope that after 
# seeing the same data multiple times the model will better determine how to estimate it. 10 ephocs -> model see the same dataset 10 times.
# We need to create something called an input function. The input function simply defines how our dataset will be converted into batches at each epoch.

# TensorFlow model we are going to use requires the data we pass it to be a tf.data.Dataset object. 
# Function converts pandas dataframe into that object.
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    # create tf.data.Dataset object with input data and its label (output data)
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)