# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:12:50 2021

@author: Carin
"""
import pandas as pd
from xgboost import XGBRegressor
import numpy as np

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
count.columns = [f'{j}_{i}' for i, j in count.columns]



sales_train = pd.merge(sales_train, count, how='left')
test = pd.merge(test, count, how='left')
#%%


cols = list(test.columns)

X_train = sales_train[cols]
y_train = sales_train['item_cnt_day']


#%%




#%%
model = XGBRegressor()
model.fit(X_train, y_train)
novel_preds= model.predict(X_val)
novel_preds= np.rint(novel_preds)

