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


#%%

count_cats = sales_train[['month_year_name', 'item_cnt_day','shop_id', 'item_category_id']]
count_cats = count_cats.groupby(['month_year_name','shop_id', 'item_category_id']).sum()
count_cats = count_cats.reset_index()
count_cats.columns = ['month_year_name', 'shop_id', 'item_category_id', 'item_category_count_per_month']

count_cats = pd.pivot_table(count_cats, values =['shop_id', 'item_category_id', 'item_category_count_per_month'], index =['shop_id', 'item_category_id'],
                         columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
count_cats.columns = [f'{j}{i}' for i, j in count_cats.columns]
count_cats.columns

#%%
grouped_train = pd.merge(grouped_train, count, how='left')
grouped_train = pd.merge(grouped_train, count_cats, how='left')
grouped_train = pd.merge(grouped_train, count_day, how='left')
test2 = pd.merge(test, count, how='left')
test2 = pd.merge(test2, count_cats, how ='left')

#%%
# cal_cols = grouped_train[['shop_id', 'item_id', 'red_day_not_sun', 'black_friday']]
# test2 = pd.merge(test2, cal_cols, how='inner')
#%%
cols =      ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month_of_year', 'June 2015item_cnt_month', 'June 2015item_category_count_per_month', 'July 2015item_cnt_month', 'July 2015item_category_count_per_month', 'August 2015item_cnt_month', 'August 2015item_category_count_per_month', 'September 2015item_cnt_month', 'September 2015item_category_count_per_month', 'October 2015item_cnt_month', 'item_cnt_month', 'month_year_name']
test_cols = ['shop_id', 'item_id', 'item_category_id', 'city_code', 'month_of_year', 'September 2015item_cnt_month', 'September 2015item_category_count_per_month', 'October 2015item_cnt_month', 'October 2015item_category_count_per_month']
test2 = test2[test_cols]

X_train = grouped_train[cols]

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

test2.rename(columns={'September 2015item_cnt_month':'lag_2', 'September 2015item_category_count_per_month':'cat_lag_2', 'October 2015item_cnt_month':'lag_1', 'October 2015item_category_count_per_month':'cat_lag_1'}, inplace=True)
#%%
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
test2.fillna(0, inplace=True)
#%%

ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_val = ss.transform(X_val)
test2 = ss.transform(test2)
#%%

model = XGBRegressor(learning_rate= 0.08661123057255304, max_depth=5, n_estimators= 810, subsample= 0.874831046, random_state=420, max_leaf_nodes= 436)
model.fit(X_train, y_train)
novel_preds= model.predict(X_val)
novel_train_preds = model.predict(X_train)
test_pred_xgboost = model.predict(test2)
novel_preds= np.rint(novel_preds)
novel_train_preds = np.rint(novel_train_preds)

MAE = mean_absolute_error(y_val, novel_preds)
MAE_Train = mean_absolute_error(y_train, novel_train_preds)
MAE
MAE_Train


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


























