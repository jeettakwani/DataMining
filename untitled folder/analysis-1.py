# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

train = pd.read_csv("data/train.csv",parse_dates=['Date'],low_memory=False)
store = pd.read_csv("data/store.csv")
test  = pd.read_csv("data/test.csv",parse_dates=['Date'],low_memory=False)

#train.head(10)

#print train.head(10)
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.countplot(x='Open',hue='DayOfWeek', data=train,palette="hls", ax=axis1)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



# Date

# Create Year and Month columns
train['Year']  = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))

test['Year']  = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))

# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)

# this column will be useful in analysis and visualization
train['Date'] = train['Date'].apply(lambda x: (str(x)[:7]))
test['Date']     = test['Date'].apply(lambda x: (str(x)[:7]))

store_id = 1
store_data = train[train["Store"] == store_id]



# group by date and get average sales, and precent change
average_sales    = store_data.groupby('Date')["Sales"].sum()
pct_change_sales = store_data.groupby('Date')["Sales"].sum().pct_change()

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(15,8))

# plot average sales over time(year-month)
ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Sales store 1")
ax1.set_xticks(range(len(average_sales)))
ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)

# Create Year and Month columns
train['Year']  = train['Date'].apply(lambda x: int(str(x)[6:]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))

test['Year']  = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))

# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)# this column will be useful in analysis and visualization
train['Date'] = train['Date'].apply(lambda x: (str(x)[:7]))
test['Date']     = test['Date'].apply(lambda x: (str(x)[:7]))

# group by date and get average sales, and precent change
average_sales    = train.groupby('Date')["Sales"].mean()
pct_change_sales = train.groupby('Date')["Sales"].sum().pct_change()

fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))

# plot average sales over time(year-month)
ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")
ax1.set_xticks(range(len(average_sales)))
ax1.set_xticklabels(average_sales.index.tolist(), rotation=90)

# plot precent change for sales over time(year-month)
ax2 = pct_change_sales.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales Percent Change")
# ax2.set_xticks(range(len(pct_change_sales)))# ax2.set_xticklabels(pct_change_sales.index.tolist(), rotation=90)

plt.show()

# plot precent change for sales over time(year-month)

#ax2 = pct_change_sales.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales Percent Change")

#ax2.set_xticks(range(len(pct_change_sales)))

#ax2.set_xticklabels(pct_change_sales.index.tolist(), rotation=90)

# .... contiune with Date

# Plot average sales & customers for every year
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Year', y='Sales', data=train, ax=axis1)
sns.barplot(x='Year', y='Customers', data=train, ax=axis2)

# Drop Date column

# train.drop(['Date'], axis=1,inplace=True)

# test.drop(['Date'], axis=1,inplace=True)

# Customers

fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,8))

# Plot max, min values, & 2nd, 3rd quartile
sns.boxplot([train["Customers"]], whis=np.inf, ax=axis1)

# group by date and get average customers, and precent change
average_customers      = train.groupby('Date')["Customers"].mean()
# pct_change_customers = train.groupby('Date')["Customers"].sum().pct_change()

# Plot average customers over the time

# it should be correlated with the average sales over time
ax = average_customers.plot(legend=True,marker='o', ax=axis2)
ax.set_xticks(range(len(average_customers)))
xlabels = ax.set_xticklabels(average_customers.index.tolist(), rotation=90)

# DayOfWeek

# In both cases where the store is closed and opened

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='DayOfWeek', y='Sales', data=train, order=[1,2,3,4,5,6,7], ax=axis1)
sns.barplot(x='DayOfWeek', y='Customers', data=train, order=[1,2,3,4,5,6,7], ax=axis2)

# Promo

# Plot average sales & customers with/without promo
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Promo', y='Sales', data=train, ax=axis1)
sns.barplot(x='Promo', y='Customers', data=train, ax=axis2)

# StateHoliday

# StateHoliday column has values 0 & "0", So, we need to merge values with 0 to "0"
train["StateHoliday"].loc[train["StateHoliday"] == 0] = "0"
# test["StateHoliday"].loc[test["StateHoliday"] == 0] = "0"

# Plot
sns.countplot(x='StateHoliday', data=train)

# Before
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StateHoliday', y='Sales', data=train, ax=axis1)

mask = (train["StateHoliday"] != "0") & (train["Sales"] > 0)
sns.barplot(x='StateHoliday', y='Sales', data=train[mask], ax=axis2)

# .... continue with StateHoliday

# After
train["StateHoliday"] = train["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
test["StateHoliday"]     = test["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StateHoliday', y='Sales', data=train, ax=axis1)
sns.barplot(x='StateHoliday', y='Customers', data=train, ax=axis2)

# SchoolHoliday

# Plot
sns.countplot(x='SchoolHoliday', data=train)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='SchoolHoliday', y='Sales', data=train, ax=axis1)
sns.barplot(x='SchoolHoliday', y='Customers', data=train, ax=axis2)

# Sales

fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,8))

# Plot max, min values, & 2nd, 3rd quartile
sns.boxplot([train["Customers"]], whis=np.inf, ax=axis1)

# Plot sales values 

# Notice that values with 0 is mostly because the store was closed
train["Sales"].plot(kind='hist',bins=70,xlim=(0,15000),ax=axis2)

# Using store

# Merge store with average store sales & customers
average_sales_customers = train.groupby('Store')[["Sales", "Customers"]].mean()
sales_customers_df = DataFrame({'Store':average_sales_customers.index,'Sales':average_sales_customers["Sales"], 'Customers': average_sales_customers["Customers"]}, 
                      columns=['Store', 'Sales', 'Customers'])
store = pd.merge(sales_customers_df, store, on='Store')

store.head()

# StoreType 

# Plot StoreType, & StoreType Vs average sales and customers

sns.countplot(x='StoreType', data=store, order=['a','b','c', 'd'])

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='StoreType', y='Sales', data=store, order=['a','b','c', 'd'],ax=axis1)
sns.barplot(x='StoreType', y='Customers', data=store, order=['a','b','c', 'd'], ax=axis2)

# Assortment 

# Plot Assortment, & Assortment Vs average sales and customers

sns.countplot(x='Assortment', data=store, order=['a','b','c'])

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Assortment', y='Sales', data=store, order=['a','b','c'], ax=axis1)
sns.barplot(x='Assortment', y='Customers', data=store, order=['a','b','c'], ax=axis2)

# Promo2

# Plot Promo2, & Promo2 Vs average sales and customers

sns.countplot(x='Promo2', data=store)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

sns.barplot(x='Promo2', y='Sales', data=store, ax=axis1)
sns.barplot(x='Promo2', y='Customers', data=store, ax=axis2)

# CompetitionDistance

# fill NaN values
store["CompetitionDistance"].fillna(store["CompetitionDistance"].median())

# Plot CompetitionDistance Vs Sales
store.plot(kind='scatter',x='CompetitionDistance',y='Sales',figsize=(15,4))
store.plot(kind='kde',x='CompetitionDistance',y='Sales',figsize=(15,4))

# What happened to the average sales of a store over time when competition started?

# Example: the average sales for store_id = 6 has dramatically decreased since the competition started

store_id = 6
store_data = train[train["Store"] == store_id]

average_store_sales = store_data.groupby('Date')["Sales"].mean()

# Get year, and month when Competition started
y = store["CompetitionOpenSinceYear"].loc[store["Store"]  == store_id].values[0]
m = store["CompetitionOpenSinceMonth"].loc[store["Store"] == store_id].values[0]

# Plot 
ax = average_store_sales.plot(legend=True,figsize=(15,4),marker='o')
ax.set_xticks(range(len(average_store_sales)))
ax.set_xticklabels(average_store_sales.index.tolist(), rotation=90)

# Since all data of store sales given in train starts with year=2013 till 2015,

# So, we need to check if year>=2013 and y & m aren't NaN values.
if y >= 2013 and y == y and m == m:
    plt.axvline(x=((y-2013) * 12) + (m - 1), linewidth=3, color='grey')

#plt.show()




plt.show()