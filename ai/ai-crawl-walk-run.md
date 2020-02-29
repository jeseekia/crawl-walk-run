# AI - Crawl, Walk, Run

## Intro
- We will save the mail carrier example for the end at which point, you should be able to attempt solving this project once we've finished the material

## Agenda
- CRAWL through some concepts
- WALK through a deeper example
- Get off and RUNning on a challenge

## Benefits:
- Reduce programming: lots of conditionals
- Customize products: provide personalized content
- Solve un-programmable problems: detect faces
- New problem-solving approach: experiments, statistics, thinking like a scientist

## Setup
Copy/save colab: https://colab.research.google.com/drive/1XnIo3Nq0UFCX66ATIqOChvpBEOOsFLXu
- Select "File"
- Select "Save a notebook in Drive"
_Uses Google Colaboratory app in Google Drive_

Training on your computer is challenging and many times impossible due to the resources needed.
Info on tools for training your model in the cloud will be included in the [resources](https://github.com/jeseekia/crawl-walk-run/blob/master/ai/ai-cwr-resources.md)

# Crawl

## Framing:

* Supervised Machine Learning Definition
Create models that combine inputs to produce useful predictions on never-before-seen data

* Label: the variable we are predicting y: spam or not spam
  - The thing you're predicting
    - Examples:
      - Price of house
      - Kind of flower in a picture
* Features: input variables describing our data ({x1, x2, ..., xn}): title of the email
  - An input variable
    - Properties of your data
  - You can have a single feature or millions of features
    - Domain Knowledge
      - Not everything is important
      - Some things are nuanced
        - Example:
          - Only using email "key phrases" can interfere with legitimate emails
    - Examples
      - Words in email text
      - Sender's address
      - Time of day the email was sent
      - Email contains the phrase "one weird trick."
* Example: an instance of the data (x): email
  * Labeled example: used to train the model: {features, label}: (x,y)
    - Includes features and the label
      - Example: label -> medianHouseValue
        | housingMedianAge | totalRooms | totalBedrooms | medianHouseValue |
        | ---------------- | ---------- | ------------- | -----------------|
        | 15 | 5612 | 1283 | 66900 |
        | 19 | 7650 | 1901 | 80100 |
        | 17 | 720 | 174 | 85700 |
        | 14 | 1501 | 337 | 73400 |
        | 20 | 1454 | 326 | 65500 |
  * Unlabeled example: used for making predictions on new data: {features, label}: (x,?)
    - Includes features but not the labels
      - Example:
        | housingMedianAge | totalRooms | totalBedrooms |
        | ---------------- | ---------- | ------------- |
        | 42 | 1686 | 361 |
        | 34 | 1226 | 180 |
        | 33 | 1077 | 271 |
* Model: maps examples to predicted labels / makes predictions: y'
  - A relationship between features and labels
  * Training
    - Creating or learning the model
    - Relationship between features and labels
  * Inference
    - Predicting or applying model to unlabeled data
    - The fun part!
  * Types of Models
    * Regression
      - Predicts continuous values
        - Example:
          - Value of a house
          - Probability a user will click on the ad
    * Classification
      - Predicts discrete values
        - Example:
          - Hotdog or Not Hotdog
          - Email is spam or not spam
          - Image of a dog, a cat, or a hamster
* Linear Regression
-
Chirps per minute vs Temp in Celsius
|       *
|  * * *      
|  *    *
|  *  *
| * *
|  *
| _ _ _ _ _ _ _ _  

Linear relationship
|      /*
|  * */*      
|  * /  *
|  */ *
| */*
| /*
|/_ _ _ _ _ _ _ _         

Slope of a line
y = mx + b
y: temp in celsius
m: slope of the line
x: chirps per minute
b: y intercept

Linear regression
y' = b + w1x1 + w2x2 + ... + wNxN
y': predicted label
b: bias (w0)
w1: weight of feature1
x1: feature 1

Infer y' by submitting a value for x1, x2, ..., xN

# Training a Model


# Loss


# Reducing Loss


# Walk

Steps to project
1. Data <-----
2. Train      |
3. Predict    |
4. Optimize   |
5. Deploy     |
6. Learn -----

Linear regression example:

// Import libraries
```
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.python.data import Dataset
```
// Load Data
```
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
```
// Format
```
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
```
// Train
  // Define features and feature columns
    ```
    my_feature = california_housing_dataframe[["total_rooms"]]
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]
    ```
  // Define target or the label to predict
    ```
    targets = california_housing_dataframe["median_house_value"]
    ```
  // Configure linear regressor
  ```
  # Use gradient descent as the optimizer for training the model.
  my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

  # Configure the linear regression model with our feature columns and optimizer.
  # Set a learning rate of 0.0000001 for Gradient Descent.
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  ```
  // Define input function to create data set
  ```
  def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
  ```
  // Train the model
  ```
  _ = linear_regressor.train(
      input_fn = lambda:my_input_fn(my_feature, targets),
      steps=100
  )
  ```
// Predict
```
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
```
// Optimize
```

```
// Deploy

## Scratching the Surface
- Don't need to know the proof behind the math. Accept the mathematic modeling and apply training based on the parameters for the model.

- Many categories of problems that you start with have some established training methods that work best

## In Practice
- People are still figuring it out
- Look for opportunities to learn from data in your company or for a project
- Experiment
  - Apply existing approaches to new problems
  - Create hybrid approaches to old problems
  - Produce something of value

## Next Steps:
Your challenge is to gather data and train a model to predict what time the mail carrier is coming.
Considerations:
- How to capture data
- What to track
- What type of model to use
- Train model
- Infer arrival time
- Bonus: Deploy model
- Bonus: Build an application for querying the model
- Bonus: Gather data through a device (camera, motion sensor, etc...)
- Bonus: Use object-detection to capture when the mail carrier arrives
- Bonus: Predict model based on visual data (camera)

## Resources:

### Algorithms
- Based on prob stats?
- Different types of problems have tuned Algorithms
| Problem | Strategy | Parameters and Return Type |
| ------- | -------- | -------------------------- |
| Predict continuous values | Linear Regression | P: feature data, training rate R: label |
| Image Recognition | ??? | ??? |
| ??? | ??? | ??? |


Skills matrix (could add math, science, other discipline headings)
| Layer | Skills Needed |
| AI training services | Problem solving |
| Using pre-trained models or "partial training" | Programming fundamentals |
| Standard "training strategies" | Programming |
| "Curated" training | Prob Stats |
| Deep Learning | Linear Algebra |
| Creating algorithms | ??? |


## Crawl (Train data to make predictions)
* Learning
  - This talk
  - Machine Learning Crash Course
* Doing
  - Train some data: Apply some training approaches to Datasets
    - Kaggle Datasets
    - Other Google results: https://www.google.com/search?q=ml+data+sets&oq=ml+data+sets&aqs=chrome..69i57j0l2.3165j0j7&sourceid=chrome&ie=UTF-8


## Walk (Deploy ML model to an application)
### Learning
  - Research papers
    - Research papers link:
    - How to read a research paper:
  - Prob Stats:
  - Linear Algebra
### Doing
  - Train Data
    - Kaggle problems
  - Complete a ML project

## Run ()
### Learning
  - Conduct Research
    - Programs
      - Open AI Scholars
    - Kaggle Datasets
### Doing
  - Solve a problem
    - Enumerate business case
    - Gather and prep data
    -

## Fly ()
### Learning
  -
### Doing
  - Create and optimize Algorithms
