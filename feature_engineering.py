import glob, re
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
import h2o
from xgboost.sklearn import XGBRegressor  
import scipy.stats as st
import matplotlib
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def feature_engineering1(hpg_reserve,store_id,air_reserve):
    hpg_reserve = pd.merge(hpg_reserve, store_id, how='inner', on=['hpg_store_id'])

    air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'])
    air_reserve['visit_datetime'] = air_reserve['visit_datetime'].dt.date
    air_reserve['reserve_datetime'] = pd.to_datetime(air_reserve['reserve_datetime'])
    air_reserve['reserve_datetime'] = air_reserve['reserve_datetime'].dt.date
    air_reserve['reserve_datetime_diff'] = air_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = air_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs_sum', 'reserve_visitors':'rv_sum'})
    tmp2 = air_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs_mean', 'reserve_visitors':'rv_mean'})
    air_reserve = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])

    hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'])
    hpg_reserve['visit_datetime'] = hpg_reserve['visit_datetime'].dt.date
    hpg_reserve['reserve_datetime'] = pd.to_datetime(hpg_reserve['reserve_datetime'])
    hpg_reserve['reserve_datetime'] = hpg_reserve['reserve_datetime'].dt.date
    hpg_reserve['reserve_datetime_diff'] = hpg_reserve.apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = hpg_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs_sum', 'reserve_visitors':'rv_sum'})
    tmp2 = hpg_reserve.groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs_mean', 'reserve_visitors':'rv_mean'})
    hpg_reserve = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
    
    reserve =pd.merge(air_reserve, hpg_reserve, how='outer', on=['air_store_id','visit_date'])
    return reserve

## 2. Create New columns - year, month, day of week, date for train/test data
def feature_engineering2(air_visit,sub,air_store):
    air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])
    air_visit['day'] = air_visit['visit_date'].dt.dayofweek
    air_visit['year'] = air_visit['visit_date'].dt.year
    air_visit['month'] = air_visit['visit_date'].dt.month
    air_visit['date'] = air_visit['visit_date'].dt.day
    air_visit = pd.merge(air_visit, air_store, how='left', on=['air_store_id']) 
   
    sub['visit_date'] = sub['id'].map(lambda x: str(x).split('_')[2])
    sub['air_store_id'] = sub['id'].map(lambda x: '_'.join(x.split('_')[:2]))
    sub['visit_date'] = pd.to_datetime(sub['visit_date'])
    sub['day'] = sub['visit_date'].dt.dayofweek
    sub['year'] = sub['visit_date'].dt.year
    sub['month'] = sub['visit_date'].dt.month
    sub['date'] = sub['visit_date'].dt.day
    sub = pd.merge(sub, air_store, how='left', on=['air_store_id']) 
    
    return(air_visit,sub,air_store)

## 3. Create New columns - min/mean/median/max/count/variation values of each store for each day of week
## LabelEncoder for genre name and area name
def feature_engineering3(sub,air_visit):

    unique_stores = sub['air_store_id'].unique()
    stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'day': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

    tmp = air_visit.groupby(['air_store_id','day'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','day']) 

    tmp = air_visit.groupby(['air_store_id','day'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','day'])

    tmp = air_visit.groupby(['air_store_id','day'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','day'])

    tmp = air_visit.groupby(['air_store_id','day'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','day'])

    tmp = air_visit.groupby(['air_store_id','day'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','day']) 

    tmp = air_visit.groupby(['air_store_id','day'], as_index=False)['visitors'].var().rename(columns={'visitors':'var_visitors'})
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','day']) 

    stores = pd.merge(stores, air_visit, how='left', on=['air_store_id','day']) 

    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
    lbl = preprocessing.LabelEncoder()
    for i in range(4):
        stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
        stores['air_area_name' +str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
    
    return(stores)

## 4. merge with holiday data
def feature_engineering4(holiday,air_visit,sub,stores,reserve):
    lbl = preprocessing.LabelEncoder()
    holiday['visit_date'] = pd.to_datetime(holiday['visit_date'])
    holiday['day_of_week'] = lbl.fit_transform(holiday['day_of_week'])

    train = pd.merge(air_visit, holiday, how='left', on=['visit_date']) 
    test = pd.merge(sub, holiday, how='left', on=['visit_date']) 
    train.drop(['latitude','longitude','air_genre_name','air_area_name'], axis=1, inplace=True)
    test.drop(['latitude','longitude','air_genre_name','air_area_name'], axis=1, inplace=True)

    ## 6. merge with all other data
    train = pd.merge(train, stores, how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, stores, how='left', on=['air_store_id','visit_date'])

    reserve['visit_date'] = pd.to_datetime(reserve['visit_date'])
    train = pd.merge(train, reserve, how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, reserve, how='left', on=['air_store_id','visit_date'])
  
    return(train,test)

## 5. Create New feature - difference with between max longitude/latitude
def feature_engineering5(train,test):
    train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)

    train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    train['var_max_lat'] = train['latitude'].max() - train['latitude']
    train['var_max_long'] = train['longitude'].max() - train['longitude']
    test['var_max_lat'] = test['latitude'].max() - test['latitude']
    test['var_max_long'] = test['longitude'].max() - test['longitude']

    lbl = preprocessing.LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])

    train = train.fillna(-999)
    test = test.fillna(-999)
    
    return(train,test)
