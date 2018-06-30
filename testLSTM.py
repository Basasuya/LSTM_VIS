# LSTM for international airline passengers problem with regression framing
import numpy
import sys
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.models import load_model
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
	look_back = 10
	for i in range(2, len(sys.argv)):
		split_dataset('srcData/' + sys.argv[i], look_back)
	scaler = MinMaxScaler(feature_range=(0, 1))
	Dataset = scaler.fit_transform(Dataset)
	model = load_model(sys.argv[1])


	prePos = 0
	for i in range(len(DataSplit)):
		for j in range(look_back+prePos, DataSplit[i]+prePos):
			trainX.append(Dataset[(j-look_back):j, 0])
			trainPos.append(j)
		prePos = prePos + DataSplit[i]

	trainX = numpy.array(trainX)
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	trainPredict = model.predict(trainX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)	
	trainPredictPlot = numpy.empty_like(Dataset)
	trainPredictPlot[:, :] = numpy.nan

	cnt = 0
	for i in trainPos:
		trainPredictPlot[i,:] = trainPredict[cnt]
		cnt = cnt + 1


	plt.plot(scaler.inverse_transform(Dataset))
	plt.plot(trainPredictPlot)
	plt.show()