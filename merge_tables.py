# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:12:50 2021

@author: Carin
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
#%%
items = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/test.csv')
shops = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/shops_en.csv', index_col = False)
item_categories = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/item_categories.csv')
calendar = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/calender.csv')

#%%
sales_train['date'] = pd.to_datetime(sales_train.date)
calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')

#%%

sales_train = pd.merge(sales_train, test, how='left')
test['month_of_year']=11

items_cats = pd.merge(items, item_categories, how='left')

sales_train = pd.merge(sales_train, items_cats, how='left')
test = pd.merge(test, items_cats, how='left')

sales_train = pd.merge(sales_train, shops, how='left')
test = pd.merge(test, shops, how='left')

sales_train = pd.merge(sales_train, calendar, how='left')
#%%
sales_train.drop(columns=['shop_id'], inplace=True)
test.drop(columns=['shop_id'], inplace=True)

sales_train.rename(columns={'unique_shop_id':'shop_id'}, inplace=True)
test.rename(columns={'unique_shop_id':'shop_id'}, inplace=True)
#%%
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
grouped_train = sales_train[['month_year_name', 'item_cnt_day', 'shop_id', 'item_id']]
grouped_train = grouped_train.groupby(['month_year_name','shop_id', 'item_id']).sum().reset_index()
grouped_train = pd.merge(grouped_train, sales_train, how='left')
#%%
grouped_train = pd.merge(grouped_train, matrix, how='outer')
del(matrix)
#%%
count = sales_train[['month_year_name', 'item_cnt_day','shop_id', 'item_id']]
count_day = count.groupby(['month_year_name','shop_id', 'item_id']).sum().reset_index()
count_day.columns = ['month_year_name', 'shop_id', 'item_id', 'item_cnt_month']

count = pd.pivot_table(count_day, values =['shop_id', 'item_id', 'item_cnt_month'], index =['shop_id', 'item_id'],
                         columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
count.columns = [f'{j}{i}' for i, j in count.columns]
count.columns
count.fillna(0, inplace=True)
count=count.astype(np.int8)

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

count_cats = sales_train[['month_year_name', 'item_cnt_day','shop_id', 'item_category_id']]
count_cats = count_cats.groupby(['month_year_name','shop_id', 'item_category_id']).sum()
count_cats = count_cats.reset_index()
count_cats.columns = ['month_year_name', 'shop_id', 'item_category_id', 'item_category_count_per_month']

count_cats = pd.pivot_table(count_cats, values =['shop_id', 'item_category_id', 'item_category_count_per_month'], index =['shop_id', 'item_category_id'],
                         columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
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
grouped_train = pd.merge(grouped_train, count, how='left')
grouped_train = pd.merge(grouped_train, count_cats, how='left')
grouped_train = pd.merge(grouped_train, count_day, how='left')
test = pd.merge(test, count, how='left')
test = pd.merge(test, count_cats, how ='left')

#%%
# cal_cols = grouped_train[['shop_id', 'item_id', 'red_day_not_sun', 'black_friday']]
# test2 = pd.merge(test2, cal_cols, how='inner')
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
cols =      ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month_of_year', 'June 2015item_cnt_month', 'June 2015item_category_count_per_month', 'July 2015item_cnt_month', 'July 2015item_category_count_per_month', 'August 2015item_cnt_month', 'August 2015item_category_count_per_month', 'September 2015item_cnt_month', 'September 2015item_category_count_per_month', 'October 2015item_cnt_month', 'item_cnt_month', 'month_year_name']
test_cols = ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month_of_year', 'September 2015item_cnt_month', 'September 2015item_category_count_per_month', 'October 2015item_cnt_month', 'October 2015item_category_count_per_month']
test = test[test_cols]

X_train = grouped_train[cols]
del(grouped_train)

X_val = X_train.loc[X_train['month_year_name'] == 'October 2015']
X_train = X_train.loc[X_train['month_year_name'] != 'October 2015']

y_train = X_train.pop('item_cnt_month')
y_val = X_val.pop('item_cnt_month')

#%%

X_val.drop(columns=['June 2015item_cnt_month', 'June 2015item_category_count_per_month', 'July 2015item_cnt_month','July 2015item_category_count_per_month', 'October 2015item_cnt_month','June 2015item_category_count_per_month', 'month_year_name'], inplace=True)

X_val.rename(columns={'August 2015item_cnt_month':'lag_2', 'September 2015item_cnt_month':'lag_1', 'August 2015item_category_count_per_month':'cat_lag_2', 'September 2015item_category_count_per_month':'cat_lag_1'}, inplace=True)

X_train.drop(columns=['September 2015item_cnt_month','September 2015item_category_count_per_month', 'October 2015item_cnt_month', 'August 2015item_cnt_month','August 2015item_category_count_per_month', 'month_year_name'], inplace=True)

X_train.rename(columns={'June 2015item_cnt_month':'lag_2', 'July 2015item_cnt_month':'lag_1', 'June 2015item_category_count_per_month':'cat_lag_2', 'July 2015item_category_count_per_month':'cat_lag_1'}, inplace=True)
#%%

test.rename(columns={'September 2015item_cnt_month':'lag_2', 'September 2015item_category_count_per_month':'cat_lag_2', 'October 2015item_cnt_month':'lag_1', 'October 2015item_category_count_per_month':'cat_lag_1'}, inplace=True)
#%%
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
test.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)
y_val.fillna(0, inplace= True)
#%%

ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_val = ss.transform(X_val)
test = ss.transform(test)
#%%

model = CatBoostRegressor(iterations=200,
                           depth=6,
                           learning_rate=0.05,
                           loss_function='RMSE',
                           verbose=True)
# train the model
model.fit(X_train, y_train)
# make the prediction using the resulting model
preds_class = model.predict(X_val)
preds_proba = model.predict_proba(X_val)
print("class = ", preds_class)
print("proba = ", preds_proba)


test_pred_xgboost=pd.DataFrame(test_pred_xgboost)
test_pred_xgboost['ID']=test['ID']
test_pred_xgboost.columns.values[0] = 'item_cnt_month'
test_pred_xgboost.columns=['item_cnt_month', 'ID']
test_pred_xgboost = test_pred_xgboost[['ID','item_cnt_month']]
test_pred_xgboost['item_cnt_month'] = test_pred_xgboost['item_cnt_month'].clip(0,20)
test_pred_xgboost.to_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/XGBoost_preds.csv', index = False)

#%%

input_layer = keras.Input(shape=(9))
x = layers.BatchNormalization()(input_layer)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation = 'relu')(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(1, activation = 'linear')(x)


model = keras.Model(inputs = input_layer, outputs = output_layer)
model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'mae',
    metrics= ['mse'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

save_best = keras.callbacks.ModelCheckpoint(
    'C:/Users/Carin/OneDrive/Documents/AW academy/Python/Miniprosjekt 2/', 
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
    batch_size=112,
    epochs=300,
    callbacks=[early_stopping, save_best],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")

history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy")



#%%
predictions = model.predict(X_train)
val_preds = model.predict(X_val)
test_pred_tf = model.predict(test2)


























