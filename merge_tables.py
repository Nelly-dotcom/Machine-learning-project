# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:12:50 2021

@author: Carin
"""
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
#%%
items = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/test.csv')
shops = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/shops_en.csv')
item_categories = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/competitive-data-science-predict-future-sales/item_categories.csv')
calendar = pd.read_csv('C:/Users/Carin/OneDrive/Documents/AW academy/Miniprosjekt 2/calender.csv')

#%%
sales_train['date'] = pd.to_datetime(sales_train.date)
calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')

#%%

sales_train = pd.merge(sales_train, test, how='left')

items_cats = pd.merge(items, item_categories, how='left')
sales_train = pd.merge(sales_train, items_cats, how='left')
test = pd.merge(test, items_cats, how='left')

sales_train = pd.merge(sales_train, shops, how='left')
test = pd.merge(test, shops, how='left')

sales_train = pd.merge(sales_train, calendar, how='left')
#%%
count = sales_train[['month_year_name', 'item_cnt_day','shop_id', 'item_id']]
count = count.groupby(['month_year_name','shop_id', 'item_id']).count()
count = count.reset_index()
count.columns = ['month_year_name', 'shop_id', 'item_id', 'item_cnt_month']

count = pd.pivot_table(count, values =['shop_id', 'item_id', 'item_cnt_month'], index =['shop_id', 'item_id'],
                         columns =['month_year_name']).reset_index().rename_axis(None, axis=0)
count.columns = [f'{j}{i}' for i, j in count.columns]
count.columns


sales_train = pd.merge(sales_train, count, how='left')
test = pd.merge(test, count, how='left')
#%%
cal_cols = sales_train[['shop_id', 'item_id', 'day_of_week', 'day_of_month', 'week', 'month_of_year', 'red_day_not_sun', 'black_friday']]
test = pd.merge(test, cal_cols, how='left')
#%%
cols = ['date', 'ID', 'shop_id', 'item_id', 'item_category_id', 'city_code', 'month_year_name', 'March 2015item_cnt_month', 'April 2015item_cnt_month', 
           'May 2015item_cnt_month', 'June 2015item_cnt_month', 'July 2015item_cnt_month', 'August 2015item_cnt_month', 'September 2015item_cnt_month', 'October 2015item_cnt_month', 'item_cnt_day']

X_train = sales_train[cols]

X_test = X_train.loc[X_train['month_year_name'] == 'October 2015']
X_train = X_train.loc[X_train['month_year_name'] != 'October 2015']

y_train = X_train.pop('item_cnt_day')
y_test = X_test.pop('item_cnt_day')

#%%

X_test.drop(columns=['March 2015item_cnt_month', 'April 2015item_cnt_month', 
           'May 2015item_cnt_month', 'June 2015item_cnt_month', 'July 2015item_cnt_month', 'October 2015item_cnt_month', 'ID', 'month_year_name', 'date'], inplace=True)

X_test.rename(columns={'August 2015item_cnt_month':'lag_2', 'September 2015item_cnt_month':'lag_1'}, inplace=True)

X_train.drop(columns=['March 2015item_cnt_month', 'April 2015item_cnt_month', 
           'May 2015item_cnt_month', 'September 2015item_cnt_month', 'October 2015item_cnt_month', 'ID', 'August 2015item_cnt_month', 'month_year_name', 'date'], inplace=True)

X_train.rename(columns={'June 2015item_cnt_month':'lag_2', 'July 2015item_cnt_month':'lag_1'}, inplace=True)
#%%

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)


#%%
model = XGBRegressor()
model.fit(X_train, y_train)
novel_preds= model.predict(X_test)
novel_preds= np.rint(novel_preds)
MAE = mean_absolute_error(y_test, novel_preds)
MAE
