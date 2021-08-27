# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:00:51 2021

@author: Nelly
"""


from itertools import product
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostRegressor, Pool

from sklearn.linear_model import LinearRegression
#%%
# Cell1
items = pd.read_csv(r'C:\Users\Nelly\Downloads\competitive-data-science-predict-future-sales\items.csv')
sales_train = pd.read_csv(r'C:\Users\Nelly\Downloads\competitive-data-science-predict-future-sales\sales_train.csv')
test = pd.read_csv(r'C:\Users\Nelly\Downloads\competitive-data-science-predict-future-sales\test.csv')
shops = pd.read_csv(r'C:\Users\Nelly\Desktop\shops_en.csv', index_col = False)
item_categories = pd.read_csv(r'C:\Users\Nelly\Desktop\item_categories_en.csv')
calendar = pd.read_csv(r'C:\Users\Nelly\Desktop\Predict prices\calender.csv')


#%%
# set date format 
sales_train['date'] = pd.to_datetime(sales_train.date)
calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')   

#%%
# merge a lot

sales_train = pd.merge(sales_train, test, how='left')
test['month_of_year']=11

items_cats = pd.merge(items, item_categories, how='left')

sales_train = pd.merge(sales_train, items_cats, how='left')
test = pd.merge(test, items_cats, how='left')

sales_train = pd.merge(sales_train, shops, how='left')
test = pd.merge(test, shops, how='left')

sales_train = pd.merge(sales_train, calendar, how='left')
#%%
# drop shop id 
sales_train.drop(columns=['shop_id'], inplace=True)
test.drop(columns=['shop_id'], inplace=True)

#give unique shop id new name
sales_train.rename(columns={'unique_shop_id':'shop_id'}, inplace=True)
test.rename(columns={'unique_shop_id':'shop_id'}, inplace=True)
#%%
# boild the structure 
build_cols = ['date_block_num','shop_id','item_id']
stack_of_months = []
for month_num in sales_train['date_block_num'].unique():
    # one month at a time, combine all combination of available shop_id and item_id
    sales = sales_train[sales_train['date_block_num']==month_num]
    this_month_unique_shop_id = sales['shop_id'].unique()
    this_month_unique_item_id = sales['item_id'].unique()
    this_month_sales = np.array(list(product(
        [month_num], this_month_unique_shop_id, this_month_unique_item_id)
        ), dtype='int16')
    stack_of_months.append(this_month_sales)

# concatenate (vstack) the list of month tables and convert it back into a dataframe
matrix = pd.DataFrame(np.vstack(stack_of_months), columns=build_cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(build_cols, inplace=True)


#%%
# make a dataframe with prefferede columns
count = sales_train[['month_year_name', 'item_cnt_day','shop_id', 'item_id', 'item_category_id']]
#group by month to get motnhly count 
count_day = count.groupby(['month_year_name','shop_id', 'item_id', 'item_category_id']).sum().reset_index()
#select the preferred columns for the dataframe
count_day.columns = ['month_year_name', 'shop_id', 'item_id', 'item_category_id', 'item_cnt_month']


count = pd.pivot_table(count_day, values =['shop_id', 'item_id', 'item_cnt_month'], index =['shop_id', 'item_id', 'item_category_id'],
                         columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
count.columns = [f'{j}{i}' for i, j in count.columns]
count.columns 

count.fillna(0, inplace=True)   #fill na with 0
#count=count.astype(np.int8)

## delete columns to make df smaller 
count.drop(columns=['April 2013item_cnt_month',
       'April 2014item_cnt_month',
       'August 2013item_cnt_month', 'August 2014item_cnt_month',
       'December 2013item_cnt_month',
       'December 2014item_cnt_month', 'February 2013item_cnt_month',
       'February 2014item_cnt_month', 
       'January 2013item_cnt_month', 'January 2014item_cnt_month',
      'July 2013item_cnt_month',
       'July 2014item_cnt_month', 
       'June 2013item_cnt_month', 'June 2014item_cnt_month',
       'March 2013item_cnt_month',
       'March 2014item_cnt_month', 
       'May 2013item_cnt_month', 'May 2014item_cnt_month',
        'November 2013item_cnt_month',
       'November 2014item_cnt_month', 
       'October 2013item_cnt_month', 'October 2014item_cnt_month',
      'September 2013item_cnt_month',
       'September 2014item_cnt_month'], inplace=True)

#%%
# It's time to count categories 
count_cats = sales_train[['month_year_name', 'item_cnt_day','shop_id', 'item_category_id']]
count_cats = count_cats.groupby(['month_year_name','shop_id', 'item_category_id']).sum()  
count_cats = count_cats.reset_index()   
count_cats.columns = ['month_year_name', 'shop_id', 'item_category_id', 'item_category_count_per_month']   #set preffered order on columns 

#pivot makes a column for each month 
count_cats = pd.pivot_table(count_cats, values =['shop_id', 'item_category_id', 'item_category_count_per_month'], index =['shop_id', 'item_category_id'],

                            columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
#get only one row of headers
count_cats.columns = [f'{j}{i}' for i, j in count_cats.columns]
count_cats.columns

count_cats.fillna(0, inplace=True)

count_cats=count_cats.drop(columns= ['April 2013item_category_count_per_month',
       'April 2014item_category_count_per_month',
       'August 2013item_category_count_per_month',
       'August 2014item_category_count_per_month',
       'December 2013item_category_count_per_month',
       'December 2014item_category_count_per_month',
       'February 2013item_category_count_per_month',
       'February 2014item_category_count_per_month',
       'January 2013item_category_count_per_month',
       'January 2014item_category_count_per_month',
       'July 2013item_category_count_per_month',
       'July 2014item_category_count_per_month',
       'June 2013item_category_count_per_month',
       'June 2014item_category_count_per_month',
       'March 2013item_category_count_per_month',
       'March 2014item_category_count_per_month',
       'May 2013item_category_count_per_month',
       'May 2014item_category_count_per_month',
       'November 2013item_category_count_per_month',
       'November 2014item_category_count_per_month',
       'October 2013item_category_count_per_month',
       'October 2014item_category_count_per_month',
       'September 2013item_category_count_per_month',
       'September 2014item_category_count_per_month'])
#%%
# including count by shop id 

date_count_cat = sales_train[['month_year_name', 'item_cnt_day', 'item_category_id']]
date_count_cat = date_count_cat.groupby(['month_year_name', 'item_category_id']).sum()
date_count_cat = date_count_cat.reset_index()
date_count_cat.columns = ['month_year_name', 'item_category_id', 'cat_count_per_month_ex_shops']

date_count_cat = pd.pivot_table(date_count_cat, values =['item_category_id', 'cat_count_per_month_ex_shops'], index =['item_category_id'],
                         columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
date_count_cat.columns = [f'{j}{i}' for i, j in date_count_cat.columns]
date_count_cat.columns

date_count_cat.fillna(0, inplace=True)

date_count_cat=date_count_cat.drop(columns= ['April 2013cat_count_per_month_ex_shops',
       'April 2014cat_count_per_month_ex_shops',
       'August 2013cat_count_per_month_ex_shops',
       'August 2014cat_count_per_month_ex_shops',
       'December 2013cat_count_per_month_ex_shops',
       'December 2014cat_count_per_month_ex_shops',
       'February 2013cat_count_per_month_ex_shops',
       'February 2014cat_count_per_month_ex_shops',
       'January 2013cat_count_per_month_ex_shops',
       'January 2014cat_count_per_month_ex_shops',
       'July 2013cat_count_per_month_ex_shops',
       'July 2014cat_count_per_month_ex_shops',
       'June 2013cat_count_per_month_ex_shops',
       'June 2014cat_count_per_month_ex_shops',
       'March 2013cat_count_per_month_ex_shops',
       'March 2014cat_count_per_month_ex_shops',
       'May 2013cat_count_per_month_ex_shops',
       'May 2014cat_count_per_month_ex_shops',
       'November 2013cat_count_per_month_ex_shops',
       'November 2014cat_count_per_month_ex_shops',
       'October 2013cat_count_per_month_ex_shops',
       'October 2014cat_count_per_month_ex_shops',
       'September 2013cat_count_per_month_ex_shops',
       'September 2014cat_count_per_month_ex_shops'])



#%% Cell9
# count_full = pd.merge(count, count_cats, how='left', left_on=['item_id', 'item_category_id'], right_on=['shop_id', 'item_category_id'])
# count_full = pd.merge(count_full, date_count_cat, how='left', left_on=['item_category_id'], right_on=['item_category_id'])

#%%
# prep train set 
grouped_train = sales_train[['month_year_name', 'item_cnt_day', 'shop_id', 'item_id', 'item_category_id']]
grouped_train = grouped_train.groupby(['month_year_name','shop_id', 'item_id', 'item_category_id']).sum().reset_index()
grouped_train = pd.merge(grouped_train, sales_train, how='left')
#%%
# outer join with the matrix
grouped_train = pd.merge(grouped_train, matrix, how='outer')
del(matrix)
#%% merge train set with the counted data 

grouped_train = pd.merge(grouped_train, count, how='left')
grouped_train = pd.merge(grouped_train, count_cats, how='left')
#grouped_train = pd.merge(grouped_train, count_full, how='inner')
grouped_train = pd.merge(grouped_train, count_day, how='left')
grouped_train = pd.merge(grouped_train, date_count_cat, how='left')

#calling it test2 to not mix up with the actual test set 
test2 = pd.merge(test, count, how='left')
test2 = pd.merge(test2, count_cats, how ='left')
test2 = pd.merge(test2, date_count_cat, how='left')

#train_head=grouped_train.head(10)  #chek the top of the set 

# cal_cols = grouped_train[['shop_id', 'item_id', 'red_day_not_sun', 'black_friday']]
# test2 = pd.merge(test2, cal_cols, how='inner')

#clean up variable explorer

del(count)
del(count_cats)
del(count_day)
del(items) 
del(sales_train)
del(shops)
del(item_categories) 
del(calendar)
del(stack_of_months)
#%%
cols =      ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month_of_year', 'June 2015item_cnt_month', 'June 2015item_category_count_per_month','June 2015cat_count_per_month_ex_shops', 'July 2015item_cnt_month', 'July 2015item_category_count_per_month','July 2015cat_count_per_month_ex_shops', 'August 2015item_cnt_month', 'August 2015item_category_count_per_month','August 2015cat_count_per_month_ex_shops', 'September 2015item_cnt_month', 'September 2015item_category_count_per_month','September 2015cat_count_per_month_ex_shops', 'October 2015item_cnt_month', 'item_cnt_month', 'month_year_name']
test_cols = ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month_of_year', 'September 2015item_cnt_month', 'September 2015item_category_count_per_month','September 2015cat_count_per_month_ex_shops', 'October 2015item_cnt_month', 'October 2015item_category_count_per_month','October 2015cat_count_per_month_ex_shops']
test2 = test2[test_cols]

X_train = grouped_train[cols]
del(grouped_train)

## split in train and val set, where November is val-set
X_val = X_train.loc[(X_train['month_year_name'] == 'October 2015') | (X_train['month_year_name'] == 'November 2015')]
X_train = X_train.loc[(X_train['month_year_name'] != 'October 2015') & (X_train['month_year_name'] != 'November 2015')]

y_train = X_train.pop('item_cnt_month')  #move item_cnt_month over to y_val with pop
y_val = X_val.pop('item_cnt_month')

#%%

X_val.drop(columns=['June 2015item_cnt_month', 'June 2015item_category_count_per_month','June 2015cat_count_per_month_ex_shops', 'July 2015item_cnt_month','July 2015item_category_count_per_month','July 2015cat_count_per_month_ex_shops', 'October 2015item_cnt_month','June 2015item_category_count_per_month', 'month_year_name'], inplace=True)

X_val.rename(columns={'August 2015item_cnt_month':'lag_2', 'September 2015item_cnt_month':'lag_1', 'August 2015item_category_count_per_month':'cat_lag_2', 'September 2015item_category_count_per_month':'cat_lag_1', 'August 2015cat_count_per_month_ex_shops':'ex_shop_cat_lag_2', 'September 2015cat_count_per_month_ex_shops':'ex_shop_cat_lag_1'}, inplace=True)

X_train.drop(columns=['September 2015item_cnt_month','September 2015item_category_count_per_month','September 2015cat_count_per_month_ex_shops', 'October 2015item_cnt_month', 'August 2015item_cnt_month','August 2015item_category_count_per_month','August 2015cat_count_per_month_ex_shops','August 2015cat_count_per_month_ex_shops', 'month_year_name'], inplace=True)

X_train.rename(columns={'June 2015item_cnt_month':'lag_2', 'July 2015item_cnt_month':'lag_1', 'June 2015item_category_count_per_month':'cat_lag_2', 'July 2015item_category_count_per_month':'cat_lag_1', 'June 2015cat_count_per_month_ex_shops':'ex_shop_cat_lag_2', 'July 2015cat_count_per_month_ex_shops':'ex_shop_cat_lag_1'}, inplace=True)

X_train.head()  #check data


## give the lag months prettier names 
test2.rename(columns={'September 2015item_cnt_month':'lag_2', 'September 2015item_category_count_per_month':'cat_lag_2', 'October 2015item_cnt_month':'lag_1', 'October 2015item_category_count_per_month':'cat_lag_1', 'September 2015cat_count_per_month_ex_shops':'ex_shop_cat_lag_1', 'October 2015cat_count_per_month_ex_shops':'ex_shop_cat_lag_2'}, inplace=True)
#%% fill onn zeroes 

X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
test2.fillna(0, inplace=True)
##
y_train.fillna(0, inplace=True)
y_val.fillna(0, inplace= True)
#%%  scale

ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_val = ss.transform(X_val)
test2 = ss.transform(test2)
#%%









model = CatBoostRegressor(iterations=300,
                           depth=4,
                           learning_rate=0.0045,
                           loss_function='RMSE',
                           verbose=True)
# train the model
model.fit(X_train, y_train)
# make the prediction using the resulting model
novel_preds = model.predict(X_val)
train_preds = model.predict(X_train)
MAE = mean_absolute_error(y_val, novel_preds)
TMAE = mean_absolute_error(y_train, train_preds)
naive_model = (y_val*0)+1
NMAE = mean_absolute_error(y_val, naive_model)
print(f'MAE: {MAE}, Naive_MAE: {NMAE}, Train_MAE: {TMAE}')

test_pred_catboost = model.predict(test2)

## save to csv 
test_pred_catboost=pd.DataFrame(test_pred_catboost)
test_pred_catboost['ID']=test['ID']
test_pred_catboost.columns.values[0] = 'item_cnt_month'
test_pred_catboost.columns=['item_cnt_month', 'ID']
test_pred_catboost = test_pred_catboost[['ID','item_cnt_month']]
test_pred_catboost['item_cnt_month'] = test_pred_catboost['item_cnt_month'].clip(0,20)
test_pred_catboost.to_csv(r'C:\Users\Nelly\Desktop\Predict prices\catfight.csv', index = False)


#%%

input_layer = keras.Input(shape=(11))
x = layers.Dense(128, activation = 'relu')(input_layer)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(1, activation = 'linear')(x)


model = keras.Model(inputs = input_layer, outputs = output_layer)
model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics= ['mae'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

save_best = keras.callbacks.ModelCheckpoint(
    r'C:\Users\Nelly\Desktop\Predict prices\mjau.csv', 
    monitor='val_loss', 
    verbose=0, 
    save_best_only=True,
    save_weights_only=False, 
    mode='auto', 
    save_freq='epoch',
    options=None
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=10,
    callbacks=[early_stopping, save_best],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="mse")



#%%
predictions = model.predict(X_train)
predictions
val_preds = model.predict(X_val)
val_preds

train_mae = mean_absolute_error(y_train, predictions)
val_mae = mean_absolute_error(y_val, val_preds)
print(f'train MAE: {train_mae}, Naive_MAE: {NMAE}, val MAE: {val_mae}')

test_pred_tf = model.predict(test2)


## linear regression 


lr_model = LinearRegression()
lr_model.fit(X=X_train, y=y_train)


# train the model
lr_model.fit(X_train, y_train)
# make the prediction using the resulting model
novel_preds = lr_model.predict(X_val)
train_preds = lr_model.predict(X_train)
MAE = mean_absolute_error(y_val, novel_preds)
TMAE = mean_absolute_error(y_train, train_preds)
naive_model = (y_val*0)+1
NMAE = mean_absolute_error(y_val, naive_model)
print(f'MAE: {MAE}, Naive_MAE: {NMAE}, Train_MAE: {TMAE}')

pred_lr = lr_model.predict(test2)

test_pred_lr=pd.DataFrame(pred_lr)
test_pred_lr['ID']=test['ID']
test_pred_lr.columns.values[0] = 'item_cnt_month'
test_pred_lr.columns=['item_cnt_month', 'ID']
test_pred_lr = test_pred_lr[['ID','item_cnt_month']]
test_pred_lr['item_cnt_month'] =test_pred_lr['item_cnt_month'].clip(0,20)
test_pred_lr.to_csv(r'C:\Users\Nelly\Desktop\Predict prices\linear_fun.csv', index = False)

import matplotlib.pyplot as plt

plt.scatter(y_train, train_preds, c='crimson')
