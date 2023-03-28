#Import Libraries
import numpy #Used for arrays and some math
import pandas #Used to access the csv for data
import tensorflow #Machine learning library
from tensorflow import keras #Specific library for machine learning
from tensorflow.keras import layers #Needed to build neural networks

stock = pandas.read_csv('SCHD_Technical_Analysis.csv') #Store stock data in variable named "stock"
stock = stock.iloc[:, 3:] #Chop off the index columns and the date column

#Define training data, testing data, and predicted values
randomized = stock.sample(len(stock), replace=False) #Shuffle the data
trainx = randomized.iloc[:round(len(stock)*2/3), :13] #Grab 2/3 of the data to train on
testx = randomized.iloc[round(len(stock)*2/3)+1:, :13] #Grab 1/3 of the data to test on
trainy = randomized.iloc[:round(len(stock)*2/3), 13] #Grab the output values of the training data
testy = randomized.iloc[round(len(stock)*2/3)+1:, 13] #Grab the output values of the test data

#Begin creating the model
model = keras.Sequential(
    [
         layers.Dense(10, activation="relu", input_shape=(13,)),
         layers.Dense(15, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(20, activation="relu"),
         layers.Dense(15, activation="relu"),
         layers.Dense(10, activation="relu"),
         layers.Dense(1)
    ]
)

#Compile the model
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsolutePercentageError()]
)

#Train and test the model
modelHistory = model.fit(
    trainx,
    trainy,
    batch_size=20,
    epochs=100,
    validation_data=(testx, testy)
)

#Print model training history to console
#print(modelHistory.history)