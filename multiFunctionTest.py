#Import Libraries
import numpy #Used for arrays and some math
import pandas #Used to access the csv for data
from tensorflow import keras #Specific library for machine learning
from tensorflow.keras import layers #Needed to build neural networks
from sklearn.model_selection import KFold #Library to help with training multiple times and finding an average result
from keras import activations #Activation functions
from keras import optimizers #Optimization functions
from xlwt import Workbook #Used to write data to an excel spreadsheet
from sklearn import preprocessing #Used to normalize data

#LeakyReLU can only be passed as a layer, so this will be used as an activation function with the same effect
def LReLU(x):
    return activations.relu(x, alpha=.1)

SCHDstock = pandas.read_csv('SCHD_Technical_Analysis.csv') #Store stock data in variable
SCHDstock = SCHDstock.iloc[:, 3:] #Chop off the index columns and the date column
SCHDstock = pandas.DataFrame(preprocessing.normalize(SCHDstock), columns=SCHDstock.columns) #Normalize the data
SPYstock = pandas.read_csv('SCHD_Technical_Analysis.csv') #Store stock data in variable
SPYstock = SPYstock.iloc[:, 3:] #Chop off the index columns and the date column
SPYstock = pandas.DataFrame(preprocessing.normalize(SPYstock), columns=SPYstock.columns) #Normalize the data
VUGstock = pandas.read_csv('SCHD_Technical_Analysis.csv') #Store stock data in variable
VUGstock = VUGstock.iloc[:, 3:] #Chop off the index columns and the date column
VUGstock = pandas.DataFrame(preprocessing.normalize(VUGstock), columns=VUGstock.columns) #Normalize the data

stocks = [SCHDstock, SPYstock, VUGstock]

kfold = KFold(n_splits=5, shuffle=True) #setup for 5-fold cross validation
finalScore = [] #array to hold the mean and std of the scores of each model

activationFunctions = [activations.relu, activations.tanh, LReLU, activations.swish, activations.gelu, activations.sigmoid]
activationNames = ["ReLU", "Tanh", "LReLU", "Swish", "GeLU", "Logistic Sigmoid"]
optimizationFunctions = [optimizers.SGD(learning_rate=.01, momentum=0.9), optimizers.Adagrad(learning_rate=.01), optimizers.RMSprop(learning_rate=.01), optimizers.Adam(learning_rate=.01)]
optimizationNames = ["SGD with Momentum", "Adagrad", "RMSprop", "Adam"]
stockNames = ["SCHD", "SPY", "VUG"]
aNameIndex = -1
oNameIndex = -1
sNameIndex = -1
row = -1
column = -1

#Create workbook to store data
wb = Workbook()

for stock in stocks:
    sNameIndex = sNameIndex + 1
    oNameIndex = -1
    
    #Define training data, testing data, and predicted values
    inData = stock.iloc[:, :13] #Define what data should be used as input
    outData = stock.iloc[:, 13:14] #Define  what data should be predicted
    inData = numpy.array(inData)
    outData = numpy.array(outData)
    
    #Create a sheet for this dataset
    sheet = wb.add_sheet(stockNames[sNameIndex])
    column = -1
    for oFunc in optimizationFunctions:
        oNameIndex = oNameIndex + 1
        aNameIndex = -1
        for aFunc in activationFunctions:
            aNameIndex = aNameIndex + 1
            errorScore = [] #Array to hold scores of each fold
            for train, test in kfold.split(inData, outData):
                #Begin creating the model
                model = keras.Sequential(
                    [
                         layers.Dense(10, activation=aFunc, input_shape=(13,)),
                         layers.Dense(15, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(20, activation=aFunc),
                         layers.Dense(15, activation=aFunc),
                         layers.Dense(10, activation=aFunc),
                         layers.Dense(1)
                    ]
                )
                
                #Compile the model
                model.compile(
                    optimizer=oFunc,
                    loss=keras.losses.MeanSquaredError(),
                    metrics=[keras.metrics.MeanAbsolutePercentageError()]
                )
                
                #Train and test the model
                modelHistory = model.fit(
                    inData[train],
                    outData[train],
                    batch_size=20,
                    epochs=500,
                    validation_data=(inData[test], outData[test])
                )
                
                #Store the score
                errorScore.append(model.metrics[1])
                
                #Store the history
                row = 0
                column = column + 1
                sheet.write(row, column, optimizationNames[oNameIndex])
                row = 1
                sheet.write(row, column, activationNames[aNameIndex])
                for data in modelHistory.history['val_mean_absolute_percentage_error']:
                    row = row + 1
                    sheet.write(row, column, data)
                
            #Convert "Mean" data type to float for calculation
            for score in range(len(errorScore)):
                errorScore[score] = errorScore[score].result().numpy()
            
            #Add the new scores to the final array
            finalScore.append([stockNames[sNameIndex], optimizationNames[oNameIndex], activationNames[aNameIndex], numpy.mean(errorScore), numpy.std(errorScore)])

#Save data to spreadsheet
wb.save("modelResults.xls")

#Print average score and std to console
for modelSet in finalScore:
    print(modelSet[0], "stock with", modelSet[1], "optimization and", modelSet[2], "activation model mean abs % error:", modelSet[3], "+/-", modelSet[4])