import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def RMSLE(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

def prep(train):
    train = train.sort_values('visit_date')
    values = np.log1p(train['visitors_x'].values).reshape(-1,1)
    values = values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    train_size = int(len(scaled) * 0.7)
    test_size = len(scaled) - train_size

    V_train, V_test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]

    look_back = 7
    trainX, trainY = create_dataset(V_train, look_back)
    testX, testY = create_dataset(V_test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    return(trainX,trainY,testX,testY)

from keras.models import Sequential
from keras.layers import Dense, LSTM
def model(trainX,trainY,testX,testY,epochs,batch_size):
	model = Sequential()
	model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')
	history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,
                            validation_data=(testX, testY), verbose=1, shuffle=False) 
	return(model)

def model_eval(model,testX,testY,train):
	scaler = MinMaxScaler(feature_range=(0, 1))
	values = np.log1p(train['visitors_x'].values).reshape(-1,1)
	values = values.astype('float32')
	scaled = scaler.fit_transform(values)

	yhat = model.predict(testX)
	yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
	testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
	rmsle = RMSLE(np.expm1(testY_inverse),np.expm1(yhat_inverse))
	print('LSTM Test RMSLE: %.3f' % rmsle)
	return(rmsle)
