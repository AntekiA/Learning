
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
dataframe = read_csv(r'C:\Users\wh110\Desktop\research\AD\real_1.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=3, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(np.shape(dataset))
test_shape = np.shape(testPredict)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+look_back: len(dataset)-1, :] = testPredict
# plot baseline and predictions
ordataset = scaler.inverse_transform(dataset)
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

y_test = ordataset
# print(y_test)
y_hat = np.empty_like(dataset)
y_hat[:, :] = np.nan
y_hat[look_back:len(trainPredict)+look_back, :] = trainPredict
y_hat[len(trainPredict)+look_back: len(dataset)-1, :] = testPredict
e = [abs(y_h - y_t[0]) for y_h,y_t in zip(y_hat, y_test)]
# plt.plot(e)
# plt.show()

smoothing_window = 5
e_s = list(pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten())
# plt.plot(e)
# plt.show()


perc_high, perc_low = np.percentile(y_test, [90, 5])
# ytest整体的高低差   和方差
inter_range = perc_high - perc_low
chan_std = np.std(y_test)
# inter_range和chan_std如上所示
# error_buffer是异常点周围被判定为异常区间的范围


def get_anomalies(window_e_s,error_buffer,inter_range,chan_std):
    mean = np.mean(window_e_s)
    sd = np.std(window_e_s)
    i_anom = []
    E_seq = []
    epsilon = mean + 2.5*sd
    # 如果太小则忽略
    if not (sd > (.05*chan_std) or max(window_e_s) > (.05 * inter_range)) or not max(window_e_s) > 0.05:
        return i_anom

    for x in range(0, len(window_e_s)):
        anom = True
        # 进行check  大于整体高低差的0.05
        if not window_e_s[x] > epsilon  or not window_e_s[x] > 0.05 * inter_range:
            anom = False

        if anom:
            for b in range(0, error_buffer):

                if not x + b in i_anom and not x + b >= len(e_s) :
                        i_anom.append(x + b)

                if not x - b in i_anom and not x - b < 0:
                        i_anom.append(x - b)
    # 进行序列转换
    i_anom = sorted(list(set(i_anom)))
    # groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
    return  i_anom


batch_size = 70
window_size = 5
# 找到窗口数目
num_windows = int((y_test.shape[0] - (batch_size * window_size)) / batch_size)
# decrease the historical error window size (h) if number of test values is limited
while num_windows <= 0:
    # 如果 windowsize过大 不断减少 找到刚好的windowsize
    window_size -= 1
    if window_size <= 0:
        window_size = 1
    num_windows = int((y_test.shape[0] - (batch_size * window_size)) / batch_size)
    # y_test长度小于batchsize
    if window_size == 1 and num_windows < 0:
        raise ValueError("Batch_size (%s) larger than y_test (len=%s). Adjust it." % (
        batch_size, y_test.shape[0]))
# print(num_windows) # 10

a = []
# 得到窗口e_s
for i in range(1, num_windows + 2):
    prior_idx = (i - 1) * (batch_size)
    # 前面有i-1个batch size
    idx = (window_size * batch_size) + ((i - 1) * batch_size)

    if i == num_windows + 1:
        # 因为最后一个加的幅度不满于config.batchsize
        idx = y_test.shape[0]
    window_e_s = e_s[prior_idx:idx]
    # print(window_e_s)
    # print(np.shape(window_e_s))
    anomalies = [i + prior_idx for i in get_anomalies(window_e_s,1,inter_range,chan_std)]
    a.extend(anomalies)
print(y_test)
print(np.shape(y_test))
print(sorted(set(a)))
