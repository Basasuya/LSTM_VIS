# LSTM for international airline passengers problem with regression framing
import numpy
import sys
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
    dataframe = read_csv(dataPath, usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    for i in dataset:
        Dataset.append(i)
    DataSplit.append(len(dataset))

if __name__ == '__main__':
	savePath = sys.argv[1]
	look_back = 10
	doTimes = int(sys.argv[2])
	for i in range(3, len(sys.argv)):
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
	model.add(LSTM(50, input_shape=(1, look_back)))
	model.add(Dense(25))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=doTimes, batch_size=1, verbose=2)
	model.save(savePath)
	