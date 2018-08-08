import numpy as np
from sklearn.ensemble import RandomForestRegressor

def RMSLE(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def rf_prep(train,test):

    col = [
        c for c in train
        if c not in ['id', 'air_store_id', 'visit_date', 'visitors_x','visitors_y']
    ]
    train = train.fillna(-1)
    test = test.fillna(-1)

    for c, dtype in zip(train.columns, train.dtypes):
        if dtype == np.float64:
            train[c] = train[c].astype(np.float32)

    for c, dtype in zip(test.columns, test.dtypes):
        if dtype == np.float64:
            test[c] = test[c].astype(np.float32)

    X=train[col]
    X_train = train[train.visit_date<'2017-03-01'][col]
    X_test = train[train.visit_date>'2017-03-01'][col]

    y_train = np.log1p(train[train.visit_date<'2017-03-01']['visitors_x'].values)
    y_test = np.log1p(train[train.visit_date>'2017-03-01']['visitors_x'].values)

    return(X,X_train,X_test,y_train,y_test)

def model_rf(X,X_train,X_test,y_train,y_test):
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    rf_model = regr.fit(X_train, y_train)

    return(rf_model)

def model_rf_eval(rf_model,X_test,y_test):
	pred=rf_model.predict(X_test)
	return(RMSLE(np.expm1(pred),np.expm1(y_test)))
