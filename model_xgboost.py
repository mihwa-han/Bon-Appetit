import numpy as np
import xgboost as xgb

def RMSLE(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5

def xgb_prep(train,test):
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
    X_valid = train[(train.visit_date>='2017-02-01') & (train.visit_date<'2017-03-01')][col]
    X_test = train[train.visit_date>'2017-03-01'][col]

    y_train = np.log1p(train[train.visit_date<'2017-03-01']['visitors_x'].values)
    y_valid = np.log1p(train[(train.visit_date>='2017-02-01') & (train.visit_date<'2017-03-01')]['visitors_x'].values)
    y_test = np.log1p(train[train.visit_date>'2017-03-01']['visitors_x'].values)

    return(X,X_train,X_valid,X_test,y_train,y_valid,y_test)

def model_xgb(X,X_train,X_valid,X_test,y_train,y_valid,y_test,xgb_params):
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=X.columns.values)
    dval = xgb.DMatrix(X_valid, y_valid, feature_names=X.columns.values)
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                               early_stopping_rounds=20, verbose_eval=10)
    return(xgb_model)

def model_xgb_eval(xgb_model,X_test,y_test):
	dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.values)
	pred=xgb_model.predict(dtest)
	return(RMSLE(np.expm1(pred),np.expm1(y_test)))
