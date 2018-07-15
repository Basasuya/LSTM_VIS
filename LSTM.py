# LSTM for international airline passengers problem with regression framing
import numpy
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
numpy.random.seed(7)

Dataset = []
DataSplit = []
trainX = []
trainY = []
testX = []
testY = []
trainPos = []
testPos = []


def split_dataset(dataPath, look_back=10):
    global Dataset, DataSplit
	#if you test sin function please use
    # dataframe = read_csv(dataPath, usecols=[1], engine='python', skipfooter=3)
    dataframe = pd.read_csv(dataPath, header=None)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    for i in dataset:
        Dataset.append(i)
    DataSplit.append(len(dataset))

if __name__ == '__main__':
	look_back = 50
	# perhaps you should modify this 
	doTimes = int(sys.argv[1])
	for i in range(2, len(sys.argv)):
		split_dataset('srcData/' + sys.argv[i], look_back)
	scaler = MinMaxScaler(feature_range=(0, 1))
	Dataset = scaler.fit_transform(Dataset)
	prePos = 0
	for i in range(len(DataSplit)):
		cnt = 0
		stopEdge = int(DataSplit[i]*0.68)
		for j in range(look_back+prePos, DataSplit[i]+prePos):
			cnt = cnt + 1
			if cnt < stopEdge:
				trainX.append(Dataset[(j-look_back):j, 0])
				trainY.append(Dataset[j, 0])
				trainPos.append(j)
			else:
				testX.append(Dataset[(j-look_back):j, 0])
				testY.append(Dataset[j, 0])
				testPos.append(j)
		prePos = prePos + DataSplit[i]

	print(len(trainX))
	trainX = numpy.array(trainX)
	trainY = numpy.array(trainY)
	testX = numpy.array(testX)
	testY = numpy.array(testY)
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(100, input_shape=(1, look_back)))
	model.add(Dense(50))
	model.add(Dense(25))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=doTimes, batch_size=1, verbose=2)
	model.save("A1.h5")

	# model = load_model("A1.h5")
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	trainPredictPlot = numpy.empty_like(Dataset)
	trainPredictPlot[:, :] = numpy.nan
	testPredictPlot = numpy.empty_like(Dataset)
	testPredictPlot[:, :] = numpy.nan

	cnt = 0
	for i in trainPos:
		trainPredictPlot[i,:] = trainPredict[cnt]
		cnt = cnt + 1

	cnt = 0
	for i in testPos:
		testPredictPlot[i,:] = testPredict[cnt]
		cnt = cnt + 1

	plt.plot(scaler.inverse_transform(Dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()