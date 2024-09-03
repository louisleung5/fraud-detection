# Fraud Detection on Financial Transactions

## Project Overview

This project aims to develop a machine learning-based fraud detection model to identify fraudulent transactions in a financial transaction datase. The project utilizes a publicly available dataset from Kaggle that contains financial transactions and their corresponding labels (whether the transaction is fraudulent or not). The dataset consists of 1,000 transactions, with 45 fraudulent transactions (4.5% of the total transactions).

The project follows a standard machine learning workflow, including explanatory data analysis, data preprocessing and feature engineering, model training and selection, and model evaluation. The following techniques are used:

1. **Explanatory data analysis**: utilize visualization technique to reveal preliminary pattern and understand the potential important features.
2. **Feature Engineering**: Extracting relevant features from (i) transaction timestamp, such as office hour indicator, weekday, Hour etc; (ii) Transaction Amount and Account Balance to create transaction amount to account balance ratio.
3. **Model Training and Selection**: Use Synthetic Minority Oversampling Technique (SMOTE) to oversample the minority class, experimenting with various machine learning algorithms, including Logistic Regression, Random Forest, and XGBoost, to identify the best-performing model.
4. **Model Evaluation**: Assessing the model's performance using recall, and interpret the results to identify the most important factors.

**Data Source**: https://www.kaggle.com/datasets/goyaladi/fraud-detection-dataset/data


```python
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
```

## 1. Import the data

Since the data files are stored in different sub-directories, I will store data in the form of dictionary, with key representing the filename and values containing data from the corresponding file.


```python
data_dir = ".\\Data\\*\\*.csv"
data_path = glob.glob(data_dir, recursive=True)
data = {}

for file in data_path:
    filename = os.path.basename(file)[:-4]
    df = pd.read_csv(file, header=0)
    data.update({filename: df})

data.keys()
```




    dict_keys(['account_activity', 'customer_data', 'fraud_indicators', 'suspicious_activity', 'merchant_data', 'transaction_category_labels', 'amount_data', 'anomaly_scores', 'transaction_metadata', 'transaction_records'])



## 2. Exploratory Data Analysis

**Transaction Data**
- transaction_records.csv: Contains transaction records with details such as transaction ID, date, amount, and customer ID. It contains 1000 transactions
- transaction_metadata.csv: Contains additional metadata for each transaction.


**Customer Profiles**
- customer_data.csv: Includes customer profiles with information such as name, age, address, and contact details. It contains 2000 customers
- account_activity.csv: Provides details of customer account activity, including account balance, last login date.


**Fraudulent Patterns**
- fraud_indicators.csv: Contains indicators of fraudulent patterns and suspicious activities.
- suspicious_activity.csv: Provides specific details of transactions flagged as suspicious.


**Transaction Amounts**
- amount_data.csv: Includes transaction amounts for each transaction.
- anomaly_scores.csv: Provides anomaly scores for transaction amounts, indicating potential fraudulence.


**Merchant Information**
- merchant_data.csv: Contains information about merchants involved in transactions.
- transaction_category_labels.csv: Provides category labels for different transaction types.

Firstly, let's combine each dataframe into one for exploration.


```python
# Transaction data
transaction = pd.merge(left=data['transaction_records'], right = data['transaction_metadata'], how = 'left', on='TransactionID')
transaction = pd.merge(left=transaction, right = data['anomaly_scores'], how = 'left', on='TransactionID')
transaction = pd.merge(left=transaction, right = data['amount_data'], how = 'left', on='TransactionID')
transaction = pd.merge(left=transaction, right = data['fraud_indicators'], how = 'left', on='TransactionID')
transaction = pd.merge(left=transaction, right = data['transaction_category_labels'], how = 'left', on='TransactionID')
transaction = pd.merge(left=transaction, right = data['merchant_data'], how = 'left', on='MerchantID')

# Customer Profiles
customer = pd.merge(left=data['customer_data'], right=data['account_activity'], on = "CustomerID" )
customer = pd.merge(left=customer, right=data['suspicious_activity'], on = "CustomerID" )

# Combine both
df = pd.merge(left=transaction, right=customer, how = "left", on = "CustomerID")
df.head(5)

# Delete the unnecessary variables to save memory
del transaction
del customer
del data
del data_dir
del data_path
del file
del filename
```

### Check for missing values and data type


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 17 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   TransactionID      1000 non-null   int64  
     1   Amount             1000 non-null   float64
     2   CustomerID         1000 non-null   int64  
     3   Timestamp          1000 non-null   object 
     4   MerchantID         1000 non-null   int64  
     5   AnomalyScore       1000 non-null   float64
     6   TransactionAmount  1000 non-null   float64
     7   FraudIndicator     1000 non-null   int64  
     8   Category           1000 non-null   object 
     9   MerchantName       1000 non-null   object 
     10  Location           1000 non-null   object 
     11  Name               1000 non-null   object 
     12  Age                1000 non-null   int64  
     13  Address            1000 non-null   object 
     14  AccountBalance     1000 non-null   float64
     15  LastLogin          1000 non-null   object 
     16  SuspiciousFlag     1000 non-null   int64  
    dtypes: float64(4), int64(6), object(7)
    memory usage: 132.9+ KB
    

#### Check for duplicated records


```python
df.duplicated().value_counts()
```




    False    1000
    Name: count, dtype: int64



It doesn't seem to have no duplicated records in the data.

#### Check for data imbalance
Given that our objective is to predict whether a transaction is fraud or not, we need to examine the proportion of our target variable (i.e., FraudIndicator) in our data.


```python
df['FraudIndicator'].value_counts()
```




    FraudIndicator
    0    955
    1     45
    Name: count, dtype: int64



It appears that only 4.5% of the transactions are classified as fraudulent, which suggests that the data is highly imbalanced.

#### Check for effectiveness of suspicious flag


```python
df.groupby('FraudIndicator')['SuspiciousFlag'].value_counts()
```




    FraudIndicator  SuspiciousFlag
    0               0                 933
                    1                  22
    1               0                  42
                    1                   3
    Name: count, dtype: int64



Based on the above, there is 45 fraud transactions, whilst only 3 transactions (i.e., 6.67%) has a suspicious flag. In addition, 22 normal transactions have a suspicious flag. This suggests that the suspicious flag may not be a perfect indicator of fraud and that there could be additional factors or features that contribute to the accurate indentification of fraudulent transaction. Also, it might be worth revising the mechanism of how the system detect suspicious activities as it does not seem to be effective at the moment.

### Data visualization

In order to get a preliminary idea on how fraudulent transaction may differ from normal transaction, it may be helpful to plot our target variable ("FraudIndicator") against other variables.


```python
nrow = 4
ncol = 4
index = 1

plt.figure(figsize=(20,15))

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='FraudIndicator', y='Amount')
index += 1

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='FraudIndicator', y='TransactionAmount')
index += 1

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='FraudIndicator', y='AnomalyScore')
index += 1

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='FraudIndicator', y='AccountBalance')
index += 1

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='FraudIndicator', y='AccountBalance')
index += 1

plt.subplot(nrow,ncol,index)
plt.bar(df['CustomerID'].unique(), df.groupby(by='CustomerID')['FraudIndicator'].sum())
plt.xlabel('CustomerID')
plt.ylabel('Count of Frauds')
index += 1

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='FraudIndicator', y='Age')
index += 1

plt.subplot(nrow,ncol,index)
plt.bar(df['MerchantID'].unique(), df.groupby(by='MerchantID')['FraudIndicator'].sum())
plt.xlabel('MerchantID')
plt.ylabel('Count of Frauds')
index += 1

plt.subplot(nrow,ncol,index)
plt.bar(df['Category'].unique(), df.groupby(by='Category')['FraudIndicator'].sum())
plt.xlabel('Transaction Category')
plt.ylabel('Count of Frauds')
index += 1

plt.subplot(nrow,ncol,index)
plt.bar(df['SuspiciousFlag'].unique(), df.groupby(by='SuspiciousFlag')['FraudIndicator'].sum(), width=0.1)
plt.xlabel('SuspiciousFlag')
plt.ylabel('Count of Frauds')
index += 1

plt.subplot(nrow,ncol,index)
sns.boxplot(data=df, x='SuspiciousFlag', y='AnomalyScore')
index += 1

plt.show()
```


    
![png](output_21_0.png)
    


Based on the above, it is observed that:
- there is no apparent distinguishment between fraudulent transaction and normal transaction, which is likely the case in real-world scenario
- Interestingly, fraudulent transactions are likely to have lower anomaly score compared to normal transaction. It is unlikely that the scoring system intends to assign a lower score to fraudulent transactions as transactions with suspicious flag tend to have a higher anamoly score. This may suggest that the scoring system is ineffective at all.
- Fraudulent transactions are likely to be under "Travel", "Online" and "Food" category.

Since the current variables do not seem to be a good predictor, it may be necessary to do some feature engineering to create some new features.

We note that there are two columns indicating the transaction amount: "TransactionAmount" and "Amount". Let's create a correlation map to see whether they are highly correlated. If yes, then they are likely to be representing the same thing and we can drop one of the column to reduce the noise.


```python
df[['TransactionAmount', 'Amount']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionAmount</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TransactionAmount</th>
      <td>1.000000</td>
      <td>-0.002585</td>
    </tr>
    <tr>
      <th>Amount</th>
      <td>-0.002585</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



It seems that they are not correlated with each other. In this case, it may worth keeping both columns for our model.

## 3. Feature Engineering

### Timestamp

Create extra columns for Month, Weekday, Day and Hour to see if fraud is likely to occur at a particular pattern in time.


```python
df.insert(4, "Month", df['Timestamp'].apply(lambda x: int(dt.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%m")))) # Get the month from "Timestamp" and show as integer
df.insert(5, "Weekday", df['Timestamp'].apply(lambda x: int(dt.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%w")))) # Get the weekday from "Timestamp" and show as integer
df.insert(5, "Day", df['Timestamp'].apply(lambda x: int(dt.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%d")))) # Get day in a month from "Timestamp" and show as integer
df.insert(6, "Hour", df['Timestamp'].apply(lambda x: int(dt.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%H")))) # Get the hour from "Timestamp" and show as integer
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>Amount</th>
      <th>CustomerID</th>
      <th>Timestamp</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Weekday</th>
      <th>MerchantID</th>
      <th>AnomalyScore</th>
      <th>...</th>
      <th>FraudIndicator</th>
      <th>Category</th>
      <th>MerchantName</th>
      <th>Location</th>
      <th>Name</th>
      <th>Age</th>
      <th>Address</th>
      <th>AccountBalance</th>
      <th>LastLogin</th>
      <th>SuspiciousFlag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>55.530334</td>
      <td>1952</td>
      <td>2022-01-01 00:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>2701</td>
      <td>0.686699</td>
      <td>...</td>
      <td>0</td>
      <td>Other</td>
      <td>Merchant 2701</td>
      <td>Location 2701</td>
      <td>Customer 1952</td>
      <td>50</td>
      <td>Address 1952</td>
      <td>2869.689912</td>
      <td>2024-08-09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>12.881180</td>
      <td>1027</td>
      <td>2022-01-01 01:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>2070</td>
      <td>0.081749</td>
      <td>...</td>
      <td>0</td>
      <td>Online</td>
      <td>Merchant 2070</td>
      <td>Location 2070</td>
      <td>Customer 1027</td>
      <td>46</td>
      <td>Address 1027</td>
      <td>9527.947107</td>
      <td>2022-01-27</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>




```python
print(f"Date range of data: {df['Timestamp'].min()} - {df['Timestamp'].max()}")
```

    Date range of data: 2022-01-01 00:00:00 - 2022-02-11 15:00:00
    


```python
nrow = 2
ncol = 2
index = 1

plt.figure(figsize=(15,10))

for var in ["Month", "Weekday", "Day", "Hour"]:
    plt.subplot(nrow,ncol,index)
    plt.bar(df[var].unique(), df.groupby(by=var)['FraudIndicator'].sum(), width=0.1)
    plt.xticks(np.sort(df[var].unique()))
    plt.xlabel(var)
    plt.ylabel('Count of Frauds')
    index += 1

plt.show()
```


    
![png](output_30_0.png)
    


Based on the above, it is observed that:
- Fraudulent transactions are likely to occur on Sunday, Tuesday and Thursday
- Most fraudulent transactions occur at 12am, 6am and 19pm, whilst there are also more fraudulent transactions during non-office hour (9am-6pm). It is unclear whether it's simply becaues there are more transactions during the non-office hour (regardless of fraudulent or not). In this case, let's create a new indicator variable of office/non-office hour.
- Month and Day may not have much insight as our data only covers January and half of february.



```python
def officehour(hour:int):
    if hour >= 9 and hour <= 18:
        return 1
    else:
        return 0

df['OfficeHour'] = df['Hour'].apply(officehour)
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>Amount</th>
      <th>CustomerID</th>
      <th>Timestamp</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Weekday</th>
      <th>MerchantID</th>
      <th>AnomalyScore</th>
      <th>...</th>
      <th>Category</th>
      <th>MerchantName</th>
      <th>Location</th>
      <th>Name</th>
      <th>Age</th>
      <th>Address</th>
      <th>AccountBalance</th>
      <th>LastLogin</th>
      <th>SuspiciousFlag</th>
      <th>OfficeHour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>55.530334</td>
      <td>1952</td>
      <td>2022-01-01 00:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>2701</td>
      <td>0.686699</td>
      <td>...</td>
      <td>Other</td>
      <td>Merchant 2701</td>
      <td>Location 2701</td>
      <td>Customer 1952</td>
      <td>50</td>
      <td>Address 1952</td>
      <td>2869.689912</td>
      <td>2024-08-09</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>12.881180</td>
      <td>1027</td>
      <td>2022-01-01 01:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>2070</td>
      <td>0.081749</td>
      <td>...</td>
      <td>Online</td>
      <td>Merchant 2070</td>
      <td>Location 2070</td>
      <td>Customer 1027</td>
      <td>46</td>
      <td>Address 1027</td>
      <td>9527.947107</td>
      <td>2022-01-27</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>50.176322</td>
      <td>1955</td>
      <td>2022-01-01 02:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>2238</td>
      <td>0.023857</td>
      <td>...</td>
      <td>Travel</td>
      <td>Merchant 2238</td>
      <td>Location 2238</td>
      <td>Customer 1955</td>
      <td>34</td>
      <td>Address 1955</td>
      <td>9288.355525</td>
      <td>2024-08-12</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>41.634001</td>
      <td>1796</td>
      <td>2022-01-01 03:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>2879</td>
      <td>0.876994</td>
      <td>...</td>
      <td>Travel</td>
      <td>Merchant 2879</td>
      <td>Location 2879</td>
      <td>Customer 1796</td>
      <td>33</td>
      <td>Address 1796</td>
      <td>5588.049942</td>
      <td>2024-03-06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>78.122853</td>
      <td>1946</td>
      <td>2022-01-01 04:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>2966</td>
      <td>0.034059</td>
      <td>...</td>
      <td>Other</td>
      <td>Merchant 2966</td>
      <td>Location 2966</td>
      <td>Customer 1946</td>
      <td>18</td>
      <td>Address 1946</td>
      <td>7324.785332</td>
      <td>2024-08-03</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df.groupby(['OfficeHour','FraudIndicator'])['TransactionID'].count().unstack(fill_value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>FraudIndicator</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>OfficeHour</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>550</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>405</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
nrow = 2
ncol = 2
index = 1

plt.figure(figsize=(15,10))

plt.subplot(nrow, ncol, index)
plt.bar(x= df['OfficeHour'].unique(), height=df.groupby('OfficeHour')['FraudIndicator'].sum())
plt.xlabel("OfficeHour Indicator")
plt.ylabel("Count of fraudulent Transactions")
plt.xticks([0,1])
index += 1

plt.subplot(nrow, ncol, index)
sns.countplot(data=df, x='OfficeHour', hue='FraudIndicator')
plt.xlabel("OfficeHour Indicator")
plt.ylabel("Count of transactions")
index += 1

plt.show()
```


    
![png](output_34_0.png)
    


### Customer data

Let's define a new column (AmtBalRatio) that represents the ratio of transaction amount to the corresponding customer's account balance.


```python
df['AmtBalRatio'] = df['TransactionAmount'] / df['AccountBalance']
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransactionID</th>
      <th>Amount</th>
      <th>CustomerID</th>
      <th>Timestamp</th>
      <th>Month</th>
      <th>Day</th>
      <th>Hour</th>
      <th>Weekday</th>
      <th>MerchantID</th>
      <th>AnomalyScore</th>
      <th>...</th>
      <th>MerchantName</th>
      <th>Location</th>
      <th>Name</th>
      <th>Age</th>
      <th>Address</th>
      <th>AccountBalance</th>
      <th>LastLogin</th>
      <th>SuspiciousFlag</th>
      <th>OfficeHour</th>
      <th>AmtBalRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>55.530334</td>
      <td>1952</td>
      <td>2022-01-01 00:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>2701</td>
      <td>0.686699</td>
      <td>...</td>
      <td>Merchant 2701</td>
      <td>Location 2701</td>
      <td>Customer 1952</td>
      <td>50</td>
      <td>Address 1952</td>
      <td>2869.689912</td>
      <td>2024-08-09</td>
      <td>0</td>
      <td>0</td>
      <td>0.027673</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>12.881180</td>
      <td>1027</td>
      <td>2022-01-01 01:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>2070</td>
      <td>0.081749</td>
      <td>...</td>
      <td>Merchant 2070</td>
      <td>Location 2070</td>
      <td>Customer 1027</td>
      <td>46</td>
      <td>Address 1027</td>
      <td>9527.947107</td>
      <td>2022-01-27</td>
      <td>0</td>
      <td>0</td>
      <td>0.001265</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>50.176322</td>
      <td>1955</td>
      <td>2022-01-01 02:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>6</td>
      <td>2238</td>
      <td>0.023857</td>
      <td>...</td>
      <td>Merchant 2238</td>
      <td>Location 2238</td>
      <td>Customer 1955</td>
      <td>34</td>
      <td>Address 1955</td>
      <td>9288.355525</td>
      <td>2024-08-12</td>
      <td>0</td>
      <td>0</td>
      <td>0.003586</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>41.634001</td>
      <td>1796</td>
      <td>2022-01-01 03:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>2879</td>
      <td>0.876994</td>
      <td>...</td>
      <td>Merchant 2879</td>
      <td>Location 2879</td>
      <td>Customer 1796</td>
      <td>33</td>
      <td>Address 1796</td>
      <td>5588.049942</td>
      <td>2024-03-06</td>
      <td>0</td>
      <td>0</td>
      <td>0.008254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>78.122853</td>
      <td>1946</td>
      <td>2022-01-01 04:00:00</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>2966</td>
      <td>0.034059</td>
      <td>...</td>
      <td>Merchant 2966</td>
      <td>Location 2966</td>
      <td>Customer 1946</td>
      <td>18</td>
      <td>Address 1946</td>
      <td>7324.785332</td>
      <td>2024-08-03</td>
      <td>0</td>
      <td>0</td>
      <td>0.007379</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
nrow = 2
ncol = 2
index = 1

plt.figure(figsize=(15,10))

plt.subplot(nrow, ncol, index)
plt.hist(df['AccountBalance'], edgecolor='black')
plt.xlabel('Account Balance')
plt.ylabel("Frequency")
index += 1

plt.subplot(nrow, ncol, index)
plt.hist(df['AmtBalRatio'], edgecolor='black')
plt.xlabel('Transaction amount to Account Balance ratio')
plt.ylabel("Frequency")
index += 1

plt.subplot(nrow, ncol, index)
sns.scatterplot(data=df, x='AmtBalRatio', y='FraudIndicator')
index += 1

plt.subplot(nrow, ncol, index)
sns.boxplot(data=df, x='FraudIndicator', y='AmtBalRatio')
index += 1

plt.show()
```


    
![png](output_37_0.png)
    


- Majority of the transactions are within 2% of the customer's account balance.
- The AmtBalRatio of fraudulent transactions are slightly less than normal transactions.

Let's also look at the address of the customer.


```python
df['Address'].unique()
```




    array(['Address 1952', 'Address 1027', 'Address 1955', 'Address 1796',
           'Address 1946', 'Address 1204', 'Address 1311', 'Address 1693',
           'Address 1347', 'Address 1574', 'Address 1424', 'Address 1302',
           'Address 1321', 'Address 1700', 'Address 1463', 'Address 1962',
           'Address 1854', 'Address 1651', 'Address 1715', 'Address 1724',
           'Address 1117', 'Address 1613', 'Address 1643', 'Address 1788',
           'Address 1251', 'Address 1960', 'Address 1187', 'Address 1121',
           'Address 1475', 'Address 1993', 'Address 1030', 'Address 1116',
           'Address 1676', 'Address 1547', 'Address 1527', 'Address 1087',
           'Address 1207', 'Address 1038', 'Address 1800', 'Address 1510',
           'Address 1104', 'Address 2000', 'Address 1644', 'Address 1238',
           'Address 1759', 'Address 1994', 'Address 1752', 'Address 1415',
           'Address 1044', 'Address 1374', 'Address 1345', 'Address 1680',
           'Address 1043', 'Address 1611', 'Address 1704', 'Address 1405',
           'Address 1161', 'Address 1336', 'Address 1714', 'Address 1865',
           'Address 1702', 'Address 1483', 'Address 1139', 'Address 1108',
           'Address 1056', 'Address 1548', 'Address 1887', 'Address 1897',
           'Address 1154', 'Address 1286', 'Address 1149', 'Address 1158',
           'Address 1606', 'Address 1422', 'Address 1007', 'Address 1985',
           'Address 1776', 'Address 1159', 'Address 1493', 'Address 1322',
           'Address 1731', 'Address 1876', 'Address 1920', 'Address 1618',
           'Address 1054', 'Address 1886', 'Address 1071', 'Address 1975',
           'Address 1726', 'Address 1435', 'Address 1684', 'Address 1277',
           'Address 1828', 'Address 1466', 'Address 1922', 'Address 1183',
           'Address 1866', 'Address 1514', 'Address 1023', 'Address 1012',
           'Address 1585', 'Address 1914', 'Address 1979', 'Address 1995',
           'Address 1359', 'Address 1609', 'Address 1685', 'Address 1961',
           'Address 1740', 'Address 1836', 'Address 1120', 'Address 1841',
           'Address 1930', 'Address 1423', 'Address 1575', 'Address 1780',
           'Address 1894', 'Address 1240', 'Address 1167', 'Address 1330',
           'Address 1171', 'Address 1749', 'Address 1069', 'Address 1569',
           'Address 1537', 'Address 1425', 'Address 1761', 'Address 1458',
           'Address 1588', 'Address 1418', 'Address 1299', 'Address 1822',
           'Address 1152', 'Address 1667', 'Address 1669', 'Address 1169',
           'Address 1019', 'Address 1017', 'Address 1150', 'Address 1129',
           'Address 1234', 'Address 1112', 'Address 1066', 'Address 1598',
           'Address 1998', 'Address 1123', 'Address 1098', 'Address 1126',
           'Address 1105', 'Address 1634', 'Address 1768', 'Address 1271',
           'Address 1935', 'Address 1666', 'Address 1495', 'Address 1982',
           'Address 1127', 'Address 1437', 'Address 1145', 'Address 1477',
           'Address 1727', 'Address 1365', 'Address 1239', 'Address 1089',
           'Address 1407', 'Address 1233', 'Address 1474', 'Address 1059',
           'Address 1596', 'Address 1084', 'Address 1591', 'Address 1107',
           'Address 1449', 'Address 1997', 'Address 1825', 'Address 1439',
           'Address 1122', 'Address 1542', 'Address 1024', 'Address 1804',
           'Address 1394', 'Address 1377', 'Address 1860', 'Address 1022',
           'Address 1599', 'Address 1281', 'Address 1549', 'Address 1237',
           'Address 1559', 'Address 1764', 'Address 1124', 'Address 1095',
           'Address 1690', 'Address 1026', 'Address 1506', 'Address 1219',
           'Address 1320', 'Address 1194', 'Address 1387', 'Address 1381',
           'Address 1292', 'Address 1556', 'Address 1616', 'Address 1646',
           'Address 1678', 'Address 1950', 'Address 1492', 'Address 1395',
           'Address 1903', 'Address 1442', 'Address 1767', 'Address 1221',
           'Address 1806', 'Address 1999', 'Address 1892', 'Address 1830',
           'Address 1515', 'Address 1479', 'Address 1398', 'Address 1848',
           'Address 1877', 'Address 1965', 'Address 1656', 'Address 1180',
           'Address 1244', 'Address 1406', 'Address 1546', 'Address 1316',
           'Address 1983', 'Address 1441', 'Address 1198', 'Address 1990',
           'Address 1893', 'Address 1558', 'Address 1253', 'Address 1455',
           'Address 1393', 'Address 1433', 'Address 1229', 'Address 1786',
           'Address 1432', 'Address 1944', 'Address 1512', 'Address 1354',
           'Address 1873', 'Address 1958', 'Address 1535', 'Address 1351',
           'Address 1562', 'Address 1817', 'Address 1649', 'Address 1505',
           'Address 1513', 'Address 1604', 'Address 1972', 'Address 1165',
           'Address 1597', 'Address 1462', 'Address 1712', 'Address 1100',
           'Address 1009', 'Address 1770', 'Address 1953', 'Address 1940',
           'Address 1135', 'Address 1701', 'Address 1392', 'Address 1468',
           'Address 1305', 'Address 1060', 'Address 1557', 'Address 1157',
           'Address 1968', 'Address 1555', 'Address 1346', 'Address 1078',
           'Address 1783', 'Address 1695', 'Address 1172', 'Address 1947',
           'Address 1086', 'Address 1275', 'Address 1657', 'Address 1624',
           'Address 1729', 'Address 1300', 'Address 1280', 'Address 1058',
           'Address 1067', 'Address 1438', 'Address 1531', 'Address 1832',
           'Address 1382', 'Address 1434', 'Address 1440', 'Address 1328',
           'Address 1957', 'Address 1033', 'Address 1615', 'Address 1137',
           'Address 1176', 'Address 1048', 'Address 1073', 'Address 1572',
           'Address 1264', 'Address 1864', 'Address 1285', 'Address 1602',
           'Address 1214', 'Address 1682', 'Address 1777', 'Address 1272',
           'Address 1427', 'Address 1339', 'Address 1855', 'Address 1967',
           'Address 1416', 'Address 1586', 'Address 1482', 'Address 1400',
           'Address 1397', 'Address 1516', 'Address 1750', 'Address 1645',
           'Address 1653', 'Address 1062', 'Address 1802', 'Address 1723',
           'Address 1146', 'Address 1807', 'Address 1540', 'Address 1709',
           'Address 1144', 'Address 1578', 'Address 1943', 'Address 1977',
           'Address 1717', 'Address 1350', 'Address 1530', 'Address 1814',
           'Address 1208', 'Address 1904', 'Address 1875', 'Address 1480',
           'Address 1282', 'Address 1109', 'Address 1304', 'Address 1906',
           'Address 1218', 'Address 1773', 'Address 1708', 'Address 1389',
           'Address 1293', 'Address 1518', 'Address 1713', 'Address 1213',
           'Address 1571', 'Address 1658', 'Address 1155', 'Address 1190',
           'Address 1175', 'Address 1376', 'Address 1018', 'Address 1608',
           'Address 1358', 'Address 1004', 'Address 1902', 'Address 1775',
           'Address 1927', 'Address 1255', 'Address 1827', 'Address 1191',
           'Address 1266', 'Address 1417', 'Address 1283', 'Address 1623',
           'Address 1488', 'Address 1008', 'Address 1497', 'Address 1696',
           'Address 1969', 'Address 1912', 'Address 1315', 'Address 1357',
           'Address 1851', 'Address 1202', 'Address 1834', 'Address 1573',
           'Address 1223', 'Address 1753', 'Address 1699', 'Address 1862',
           'Address 1197', 'Address 1626', 'Address 1931', 'Address 1343',
           'Address 1587', 'Address 1276', 'Address 1484', 'Address 1655',
           'Address 1025', 'Address 1794', 'Address 1295', 'Address 1741',
           'Address 1600', 'Address 1174', 'Address 1792', 'Address 1265',
           'Address 1360', 'Address 1356', 'Address 1352', 'Address 1697',
           'Address 1050', 'Address 1318', 'Address 1503', 'Address 1029',
           'Address 1052', 'Address 1111', 'Address 1501', 'Address 1231',
           'Address 1978', 'Address 1496', 'Address 1629', 'Address 1837',
           'Address 1954', 'Address 1016', 'Address 1460', 'Address 1660',
           'Address 1260', 'Address 1507', 'Address 1119', 'Address 1141',
           'Address 1307', 'Address 1289', 'Address 1923', 'Address 1668',
           'Address 1732', 'Address 1487', 'Address 1102', 'Address 1799',
           'Address 1125', 'Address 1051', 'Address 1268', 'Address 1882',
           'Address 1751', 'Address 1769', 'Address 1853', 'Address 1622',
           'Address 1412', 'Address 1036', 'Address 1665', 'Address 1096',
           'Address 1367', 'Address 1340', 'Address 1973', 'Address 1005',
           'Address 1565', 'Address 1691', 'Address 1470', 'Address 1755',
           'Address 1132', 'Address 1414', 'Address 1639', 'Address 1845',
           'Address 1821', 'Address 1595', 'Address 1686', 'Address 1901',
           'Address 1730', 'Address 1758', 'Address 1385', 'Address 1263',
           'Address 1736', 'Address 1399', 'Address 1612', 'Address 1303',
           'Address 1147', 'Address 1469', 'Address 1210', 'Address 1003',
           'Address 1077', 'Address 1309', 'Address 1287', 'Address 1905',
           'Address 1561', 'Address 1908', 'Address 1279', 'Address 1926',
           'Address 1519', 'Address 1216', 'Address 1142', 'Address 1937',
           'Address 1200', 'Address 1103', 'Address 1652', 'Address 1694',
           'Address 1739', 'Address 1201', 'Address 1733', 'Address 1971',
           'Address 1430', 'Address 1148', 'Address 1976', 'Address 1879',
           'Address 1428', 'Address 1683', 'Address 1570', 'Address 1189',
           'Address 1637', 'Address 1581', 'Address 1835', 'Address 1072',
           'Address 1045', 'Address 1989', 'Address 1410', 'Address 1401',
           'Address 1898', 'Address 1891', 'Address 1348', 'Address 1090',
           'Address 1368', 'Address 1743', 'Address 1366', 'Address 1610',
           'Address 1473', 'Address 1630', 'Address 1413', 'Address 1521',
           'Address 1252', 'Address 1269', 'Address 1143', 'Address 1094',
           'Address 1577', 'Address 1520', 'Address 1160', 'Address 1057',
           'Address 1168', 'Address 1711', 'Address 1671', 'Address 1163',
           'Address 1722', 'Address 1532', 'Address 1464', 'Address 1091',
           'Address 1182', 'Address 1771', 'Address 1049', 'Address 1500',
           'Address 1001', 'Address 1986', 'Address 1921', 'Address 1808',
           'Address 1195', 'Address 1673', 'Address 1528', 'Address 1429',
           'Address 1273', 'Address 1177', 'Address 1419', 'Address 1924',
           'Address 1843', 'Address 1625', 'Address 1899', 'Address 1687',
           'Address 1981', 'Address 1991', 'Address 1205', 'Address 1744',
           'Address 1579', 'Address 1386', 'Address 1296', 'Address 1490',
           'Address 1647', 'Address 1188', 'Address 1053', 'Address 1454',
           'Address 1551', 'Address 1934', 'Address 1980', 'Address 1765',
           'Address 1553', 'Address 1839', 'Address 1249', 'Address 1128',
           'Address 1085', 'Address 1212', 'Address 1917', 'Address 1504',
           'Address 1928', 'Address 1533', 'Address 1319', 'Address 1544',
           'Address 1538', 'Address 1996', 'Address 1225', 'Address 1932',
           'Address 1992', 'Address 1228', 'Address 1747', 'Address 1199',
           'Address 1227', 'Address 1334', 'Address 1242', 'Address 1136',
           'Address 1785', 'Address 1801', 'Address 1088', 'Address 1705',
           'Address 1131', 'Address 1232', 'Address 1014', 'Address 1779',
           'Address 1568', 'Address 1083', 'Address 1138', 'Address 1942',
           'Address 1592', 'Address 1037', 'Address 1748', 'Address 1925',
           'Address 1631', 'Address 1628', 'Address 1951', 'Address 1511',
           'Address 1661', 'Address 1787', 'Address 1153', 'Address 1055',
           'Address 1020', 'Address 1421', 'Address 1884', 'Address 1706',
           'Address 1312', 'Address 1411', 'Address 1566', 'Address 1654'],
          dtype=object)




```python
df.groupby('CustomerID')['Address'].nunique().sort_values(ascending=False)
```




    CustomerID
    2000    1
    1001    1
    1003    1
    1004    1
    1005    1
           ..
    1050    1
    1051    1
    1052    1
    1053    1
    1054    1
    Name: Address, Length: 636, dtype: int64



In real-life scenario, it may be possible that fraudulent transactions is more likely to arise from one location than another, and we can extract the country/city/districts from the address and use those features in our training. 

However, in our case it appears that the addresses of the customers are encoded (potentially for confidentiality reason) and each customer is assigned with a unique address ID. Therefore, address may not give us any indication on fraudulent transactions and hence we will not use it in our training.

Let's also look at the difference between transaction date and last login. 


```python
print(f"Last Login date range: {df['LastLogin'].min()} - {df['LastLogin'].max()}") 
```

    Last Login date range: 2022-01-01 - 2024-09-26
    


```python
df[['Timestamp', 'LastLogin']].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>LastLogin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-01-01 00:00:00</td>
      <td>2024-08-09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-01-01 01:00:00</td>
      <td>2022-01-27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-01-01 02:00:00</td>
      <td>2024-08-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-01-01 03:00:00</td>
      <td>2024-03-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-01-01 04:00:00</td>
      <td>2024-08-03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022-01-01 05:00:00</td>
      <td>2022-07-23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2022-01-01 06:00:00</td>
      <td>2022-11-07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2022-01-01 07:00:00</td>
      <td>2023-11-24</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022-01-01 08:00:00</td>
      <td>2022-12-13</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2022-01-01 09:00:00</td>
      <td>2023-07-28</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2022-01-01 10:00:00</td>
      <td>2023-02-28</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2022-01-01 11:00:00</td>
      <td>2022-10-29</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2022-01-01 12:00:00</td>
      <td>2022-11-17</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2022-01-01 13:00:00</td>
      <td>2023-12-01</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2022-01-01 14:00:00</td>
      <td>2022-11-17</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2022-01-01 15:00:00</td>
      <td>2023-04-08</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2022-01-01 16:00:00</td>
      <td>2024-08-19</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2022-01-01 17:00:00</td>
      <td>2024-05-03</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2022-01-01 18:00:00</td>
      <td>2023-10-13</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2022-01-01 19:00:00</td>
      <td>2023-12-16</td>
    </tr>
  </tbody>
</table>
</div>



The Last Login dates for most records are later than the transaction dates, suggesting that the data was collected at a point in time after the actual transactions occurred, whilst `LastLogin` represents the last login date as at the data collection date, instead of the last login date prior to the transaction.

In case where we would like to make immediate prediction as soon as the transaction occurs, `LastLogin` may not give valuable insights because it will just simply be same as the transaction date. As a result, we will not include this feature in our model.

## 4. Feature Selection and data preprocessing

Based on the above explanatory data analysis, let's select the following features for training our model:
- Amount
- TransactionAmount
- Weekday
- OfficeHour
- AmtBalRatio
- AnomalyScore
- SuspiciousFlag
- Category


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 23 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   TransactionID      1000 non-null   int64  
     1   Amount             1000 non-null   float64
     2   CustomerID         1000 non-null   int64  
     3   Timestamp          1000 non-null   object 
     4   Month              1000 non-null   int64  
     5   Day                1000 non-null   int64  
     6   Hour               1000 non-null   int64  
     7   Weekday            1000 non-null   int64  
     8   MerchantID         1000 non-null   int64  
     9   AnomalyScore       1000 non-null   float64
     10  TransactionAmount  1000 non-null   float64
     11  FraudIndicator     1000 non-null   int64  
     12  Category           1000 non-null   object 
     13  MerchantName       1000 non-null   object 
     14  Location           1000 non-null   object 
     15  Name               1000 non-null   object 
     16  Age                1000 non-null   int64  
     17  Address            1000 non-null   object 
     18  AccountBalance     1000 non-null   float64
     19  LastLogin          1000 non-null   object 
     20  SuspiciousFlag     1000 non-null   int64  
     21  OfficeHour         1000 non-null   int64  
     22  AmtBalRatio        1000 non-null   float64
    dtypes: float64(5), int64(11), object(7)
    memory usage: 179.8+ KB
    


```python
X = df[['Amount', 'TransactionAmount', 'Weekday', 'OfficeHour', 'AmtBalRatio', 'AnomalyScore', 'SuspiciousFlag', 'Category']].copy(deep=True)
y = df['FraudIndicator'].copy(deep=True)
```

Before we can train our model, we need to first encode our categorical feature into binary indicator. Since logistic regression will be affected by multi-collinearity, we should drop one of the category. In this case, we will create two sets of data: "lr" data for logistic regression and "tr" for tree models.


```python
X_lr = pd.get_dummies(X, columns=['Category'], drop_first=True)
X_tr = pd.get_dummies(X, columns=['Category'], drop_first=False)
```

In addition, we should also normalize our numerical data.


```python
scaler = StandardScaler()
numerical_features = ['Amount', 'TransactionAmount']

for feature in numerical_features:
    X_lr[feature] = scaler.fit_transform(X_lr[[feature]])
    X_tr[feature] = scaler.fit_transform(X_tr[[feature]])

print(X_lr.describe())
print(X_tr.describe())
```

                 Amount  TransactionAmount      Weekday  OfficeHour  AmtBalRatio  \
    count  1.000000e+03       1.000000e+03  1000.000000  1000.00000  1000.000000   
    mean   1.598721e-16       3.907985e-17     2.984000     0.41700     0.013164   
    std    1.000500e+00       1.000500e+00     2.000937     0.49331     0.011948   
    min   -1.811296e+00      -1.755943e+00     0.000000     0.00000     0.001085   
    25%   -8.337589e-01      -8.427235e-01     1.000000     0.00000     0.006013   
    50%    9.775030e-02       4.006338e-03     3.000000     0.00000     0.009758   
    75%    8.166447e-01       8.334011e-01     5.000000     1.00000     0.015339   
    max    1.775563e+00       1.684416e+00     6.000000     1.00000     0.077885   
    
           AnomalyScore  SuspiciousFlag  
    count   1000.000000     1000.000000  
    mean       0.492282        0.025000  
    std        0.288423        0.156203  
    min        0.000234        0.000000  
    25%        0.251802        0.000000  
    50%        0.490242        0.000000  
    75%        0.741888        0.000000  
    max        0.999047        1.000000  
                 Amount  TransactionAmount      Weekday  OfficeHour  AmtBalRatio  \
    count  1.000000e+03       1.000000e+03  1000.000000  1000.00000  1000.000000   
    mean   1.598721e-16       3.907985e-17     2.984000     0.41700     0.013164   
    std    1.000500e+00       1.000500e+00     2.000937     0.49331     0.011948   
    min   -1.811296e+00      -1.755943e+00     0.000000     0.00000     0.001085   
    25%   -8.337589e-01      -8.427235e-01     1.000000     0.00000     0.006013   
    50%    9.775030e-02       4.006338e-03     3.000000     0.00000     0.009758   
    75%    8.166447e-01       8.334011e-01     5.000000     1.00000     0.015339   
    max    1.775563e+00       1.684416e+00     6.000000     1.00000     0.077885   
    
           AnomalyScore  SuspiciousFlag  
    count   1000.000000     1000.000000  
    mean       0.492282        0.025000  
    std        0.288423        0.156203  
    min        0.000234        0.000000  
    25%        0.251802        0.000000  
    50%        0.490242        0.000000  
    75%        0.741888        0.000000  
    max        0.999047        1.000000  
    


```python
X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y, test_size=0.2, random_state=1207, stratify=y)
X_tr_train, X_tr_test, y_tr_train, y_tr_test = train_test_split(X_tr, y, test_size=0.2, random_state=1207, stratify=y)

len(X_lr_train), len(X_lr_test), len(y_lr_train), len(y_lr_test), len(X_tr_train), len(X_tr_test), len(y_tr_train), len(y_tr_test)
```




    (800, 200, 800, 200, 800, 200, 800, 200)



Since our data is highly imbalanced, we need to use SMOTE to oversample the minority class in our training data. Note that we should never use such technique on our testing data because the testing data is supposed to validate our performance under a real-life situation.


```python
sm = SMOTE(sampling_strategy=0.5, random_state=1207)
X_lr_train, y_lr_train = sm.fit_resample(X=X_lr_train, y=y_lr_train)
X_tr_train, y_tr_train = sm.fit_resample(X=X_tr_train, y=y_tr_train)

len(X_lr_train), len(y_lr_train), len(X_tr_train), len(y_tr_train)
```




    (1146, 1146, 1146, 1146)




```python
print(y_lr_train.value_counts())
print(y_tr_train.value_counts())
```

    FraudIndicator
    0    764
    1    382
    Name: count, dtype: int64
    FraudIndicator
    0    764
    1    382
    Name: count, dtype: int64
    

## 5. Model Training

Let's define a function for model revaluation and storing the result.


```python
def make_results(model_name:str, y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    y_pred_proba = [y_pred_proba[x][1] for x in range(len(y_pred_proba))] # Transform the prediction probability into a list
    auroc = roc_auc_score(y_true=y_true, y_score=y_pred_proba)
    prec, rec, thres = precision_recall_curve(y_true, y_pred_proba)
    auprc = auc(rec, prec)
    result = pd.DataFrame({'Model': [model_name],
                        'F1': [round(f1,4)],
                        'Recall': [round(recall,4)],
                        'Precision': [round(precision,4)],
                        'Accuracy': [round(accuracy,4)],
                        'AArea under ROC curve': [round(auroc, 4)],
                        'Area under Precision-recall Curve': [round(auprc, 4)]})
    return result

def update_result_table(existing_result_table: pd.DataFrame, new_result_table: pd.DataFrame):
    return pd.concat([existing_result_table, new_result_table], axis=0, ignore_index=True)
```

### Logistics Regression


```python
lr = LogisticRegression(penalty='l2', random_state=1207)
lr = lr.fit(X_lr_train, y_lr_train)

y_lr_train_pred = lr.predict(X_lr_train).tolist()
y_lr_train_pred_proba = lr.predict_proba(X_lr_train).tolist()
lr_train_result = make_results(model_name="Logistic Regression_Train", y_true=y_lr_train, y_pred=y_lr_train_pred, y_pred_proba=y_lr_train_pred_proba)


y_lr_test_pred = lr.predict(X_lr_test).tolist()
y_lr_test_pred_proba = lr.predict_proba(X_lr_test).tolist()
lr_test_result = make_results(model_name="Logistic Regression_Test", y_true=y_lr_test, y_pred=y_lr_test_pred, y_pred_proba=y_lr_test_pred_proba)
```


```python
result = update_result_table(lr_train_result, lr_test_result)
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Accuracy</th>
      <th>AArea under ROC curve</th>
      <th>Area under Precision-recall Curve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression_Train</td>
      <td>0.7504</td>
      <td>0.6492</td>
      <td>0.8889</td>
      <td>0.856</td>
      <td>0.9106</td>
      <td>0.8771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression_Test</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.915</td>
      <td>0.4171</td>
      <td>0.0426</td>
    </tr>
  </tbody>
</table>
</div>



### Decision Tree

In order to perform hyperparameter tuning, we will use GridSearch cross-validation technique to look for the optimal combination for the tree model. Bear in mind that our objective is to detect fraudulent transactions which are considerably rare in reality, and the consequence of mis-classifying a normal transcation as fraudulent transaction is much smaller than mis-classifying a fraudulent transaction as normal transaction. Therefore, our scoring metric should be focused on recall.


```python
# Define the dictionary of hyperparameter values
tree_parameter = {'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50],
                  'min_samples_leaf':[2,5,10,20,50]}

# Define the metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']

decision_tree = DecisionTreeClassifier(random_state=1207)
dt_cv = GridSearchCV(decision_tree,
                     tree_parameter,
                     scoring=scoring,
                     cv=5, # The number of cross-validation folds
                     refit='recall').fit(X_tr_train, y_tr_train) # The scoring metric that you want GridSearch to use when it selects the "best" model. The reason it's called refit is because once the algorithm finds the combination of hyperparameters that results in the best average score across all validation folds, it will then refit this model to all of the training data. Remember, up until now, with a 5-fold cross-validation, the model has only ever been fit on 80% (4/5) of the training data, because the remaining 20% was held out as a validation fold.)
print(f"Best model estimator: {dt_cv.best_estimator_}")
print(f"Best Avg. Validation Score: {dt_cv.best_score_:.4f}%")

y_tr_train_pred = dt_cv.predict(X_tr_train)
y_tr_train_pred_proba = dt_cv.predict_proba(X_tr_train)
decision_tree_train_result = make_results(model_name="Decision Tree_Train", y_true = y_tr_train, y_pred = y_tr_train_pred, y_pred_proba=y_tr_train_pred_proba)
result = update_result_table(result, decision_tree_train_result)

y_tr_test_pred = dt_cv.predict(X_tr_test)
y_tr_test_pred_proba = dt_cv.predict_proba(X_tr_test)
decision_tree_test_result = make_results(model_name="Decision Tree_Test", y_true = y_tr_test, y_pred = y_tr_test_pred, y_pred_proba=y_tr_test_pred_proba)
result = update_result_table(result, decision_tree_test_result)
```

    Best model estimator: DecisionTreeClassifier(max_depth=11, min_samples_leaf=2, random_state=1207)
    Best Avg. Validation Score: 0.8198%
    

### Random Forest

Similar to decision tree, GridSearch cross-validation technique will be used to look for the optimal combination of the random forest model.


```python
import warnings

warnings.filterwarnings(action='ignore')
```


```python
rf_params = {'max_depth': [2,3,4,5,None], # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
             'min_samples_leaf': [1,2,3], # The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
             'min_samples_split': [2,3,4], # The minimum number of samples required to split an internal node
             'max_features': [2,3,4], # The number of features to consider when looking for the best split
             'n_estimators': [75, 100, 125, 150]} # The number of trees in the forest.

# Define the metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']

rf = RandomForestClassifier(random_state=1207)
rf_cv = GridSearchCV(rf,
                     rf_params,
                     scoring=scoring,
                     cv=5, # The number of cross-validation folds
                     refit='recall').fit(X_tr_train, y_tr_train) 

print(f"Best model estimator: {rf_cv.best_estimator_}")
print(f"Best Avg. Validation Score: {rf_cv.best_score_:.4f}%")

y_tr_train_pred = rf_cv.predict(X_tr_train)
y_tr_train_pred_proba = rf_cv.predict_proba(X_tr_train)
rf_train_result = make_results(model_name="Random Forest_Train", y_true = y_tr_train, y_pred = y_tr_train_pred, y_pred_proba=y_tr_train_pred_proba)
result = update_result_table(result, rf_train_result)

y_tr_test_pred = rf_cv.predict(X_tr_test)
y_tr_test_pred_proba = rf_cv.predict_proba(X_tr_test)
rf_test_result = make_results(model_name="Random Forest_Test", y_true = y_tr_test, y_pred = y_tr_test_pred, y_pred_proba=y_tr_test_pred_proba)
result = update_result_table(result, rf_test_result)
result

```

    Best model estimator: RandomForestClassifier(max_features=2, min_samples_split=3, n_estimators=150,
                           random_state=1207)
    Best Avg. Validation Score: 0.9297%
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Accuracy</th>
      <th>AArea under ROC curve</th>
      <th>Area under Precision-recall Curve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression_Train</td>
      <td>0.7504</td>
      <td>0.6492</td>
      <td>0.8889</td>
      <td>0.8560</td>
      <td>0.9106</td>
      <td>0.8771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression_Test</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.9150</td>
      <td>0.4171</td>
      <td>0.0426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree_Train</td>
      <td>0.9499</td>
      <td>0.9424</td>
      <td>0.9574</td>
      <td>0.9668</td>
      <td>0.9966</td>
      <td>0.9934</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Decision Tree_Test</td>
      <td>0.2500</td>
      <td>0.4444</td>
      <td>0.1739</td>
      <td>0.8800</td>
      <td>0.6521</td>
      <td>0.2264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest_Train</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest_Test</td>
      <td>0.1818</td>
      <td>0.1111</td>
      <td>0.5000</td>
      <td>0.9550</td>
      <td>0.4316</td>
      <td>0.1652</td>
    </tr>
  </tbody>
</table>
</div>



### XGBoost


```python
xgb_params = {'max_depth': [4,5,6,7,8], # Specifies how many levels your base learner trees can have. 
             'min_child_weight': [1,2,3,4,5], # Controls threshold below which a node becomes a leaf, based on the combined weight of the samples it contains.  For regression models, this value is functionally equivalent to a number of samples. For the binary classification objective, the weight of a sample in a node is dependent on its probability of response as calculated by that tree. The weight of the sample decreases the more certain the model is (i.e., the closer the probability of response is to 0 or 1).
             'learning_rate': [0.1,0.2,0.3], # Controls how much importance is given to each consecutive base learner in the ensemble’s final prediction. Also known as eta or shrinkage. 
             'n_estimators': [75, 100, 125]} # Specifies the number of boosting rounds (i.e., the number of trees your model will build in its ensemble)

xgb = XGBClassifier(objective='binary:logistic', random_state=1207)

scoring = ['accuracy', 'precision', 'recall', 'f1']

xgb_cv = GridSearchCV(xgb, xgb_params, scoring=scoring, cv=5, refit='recall').fit(X_tr_train, y_tr_train)

print(f"Best model estimator: {xgb_cv.best_estimator_}")
print(f"Best Avg. Validation Score: {xgb_cv.best_score_:.4f}%")

y_tr_train_pred = xgb_cv.predict(X_tr_train)
y_tr_train_pred_proba = xgb_cv.predict_proba(X_tr_train)
xgb_train_result = make_results(model_name="XGBoost_Train", y_true = y_tr_train, y_pred = y_tr_train_pred, y_pred_proba = y_tr_train_pred_proba)
result = update_result_table(result, xgb_train_result)

y_tr_test_pred = xgb_cv.predict(X_tr_test)
y_tr_test_pred_proba = xgb_cv.predict_proba(X_tr_test)
xgb_test_result = make_results(model_name="XGBoost_Test", y_true = y_tr_test, y_pred = y_tr_test_pred, y_pred_proba = y_tr_test_pred_proba)
result = update_result_table(result, xgb_test_result)
result
```

    Best model estimator: XGBClassifier(base_score=None, booster=None, callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, device=None, early_stopping_rounds=None,
                  enable_categorical=False, eval_metric=None, feature_types=None,
                  gamma=None, grow_policy=None, importance_type=None,
                  interaction_constraints=None, learning_rate=0.3, max_bin=None,
                  max_cat_threshold=None, max_cat_to_onehot=None,
                  max_delta_step=None, max_depth=7, max_leaves=None,
                  min_child_weight=1, missing=nan, monotone_constraints=None,
                  multi_strategy=None, n_estimators=125, n_jobs=None,
                  num_parallel_tree=None, random_state=1207, ...)
    Best Avg. Validation Score: 0.9350%
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Accuracy</th>
      <th>AArea under ROC curve</th>
      <th>Area under Precision-recall Curve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression_Train</td>
      <td>0.7504</td>
      <td>0.6492</td>
      <td>0.8889</td>
      <td>0.8560</td>
      <td>0.9106</td>
      <td>0.8771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression_Test</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.9150</td>
      <td>0.4171</td>
      <td>0.0426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree_Train</td>
      <td>0.9499</td>
      <td>0.9424</td>
      <td>0.9574</td>
      <td>0.9668</td>
      <td>0.9966</td>
      <td>0.9934</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Decision Tree_Test</td>
      <td>0.2500</td>
      <td>0.4444</td>
      <td>0.1739</td>
      <td>0.8800</td>
      <td>0.6521</td>
      <td>0.2264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest_Train</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest_Test</td>
      <td>0.1818</td>
      <td>0.1111</td>
      <td>0.5000</td>
      <td>0.9550</td>
      <td>0.4316</td>
      <td>0.1652</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost_Train</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>XGBoost_Test</td>
      <td>0.1333</td>
      <td>0.1111</td>
      <td>0.1667</td>
      <td>0.9350</td>
      <td>0.5608</td>
      <td>0.1658</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_importance(xgb_cv.best_estimator_)
```




    <Axes: title={'center': 'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_74_1.png)
    



```python
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1</th>
      <th>Recall</th>
      <th>Precision</th>
      <th>Accuracy</th>
      <th>AArea under ROC curve</th>
      <th>Area under Precision-recall Curve</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression_Train</td>
      <td>0.7504</td>
      <td>0.6492</td>
      <td>0.8889</td>
      <td>0.8560</td>
      <td>0.9106</td>
      <td>0.8771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression_Test</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.9150</td>
      <td>0.4171</td>
      <td>0.0426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree_Train</td>
      <td>0.9499</td>
      <td>0.9424</td>
      <td>0.9574</td>
      <td>0.9668</td>
      <td>0.9966</td>
      <td>0.9934</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Decision Tree_Test</td>
      <td>0.2500</td>
      <td>0.4444</td>
      <td>0.1739</td>
      <td>0.8800</td>
      <td>0.6521</td>
      <td>0.2264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest_Train</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest_Test</td>
      <td>0.1818</td>
      <td>0.1111</td>
      <td>0.5000</td>
      <td>0.9550</td>
      <td>0.4316</td>
      <td>0.1652</td>
    </tr>
    <tr>
      <th>6</th>
      <td>XGBoost_Train</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>XGBoost_Test</td>
      <td>0.1333</td>
      <td>0.1111</td>
      <td>0.1667</td>
      <td>0.9350</td>
      <td>0.5608</td>
      <td>0.1658</td>
    </tr>
  </tbody>
</table>
</div>



## 6. Model Evaluation

Based on the performance on the testing set, it appears that our decision tree model has the highest recall score, hence it is regarded as our best performing model so far. Now, we can plot the decision tree and try to interpret the result.

Note that the performance on the testing sets seems to be significantly different from the performance on training sets for all models, indicating potential overfitting.


```python
plt.figure(figsize=(20,10))
plot_tree(dt_cv.best_estimator_, max_depth=4, fontsize=8, feature_names=X_tr_train.columns, class_names={0:'normal', 1:'fraudulent'}, filled=True)
plt.show()
```


    
![png](output_78_0.png)
    


As observed during explanatory data analysis, it appears that office hour appears to be the first decision factor, followed by transaction category (whether it is an online/travel transaction), transaction amount and Anomaly score.


```python
feature_importance = sorted(list(zip(dt_cv.best_estimator_.feature_names_in_,dt_cv.best_estimator_.feature_importances_.tolist())), key=lambda x: x[1], reverse=True)
feature_importance
```




    [('Amount', 0.1384003612498369),
     ('OfficeHour', 0.12888639304173805),
     ('AmtBalRatio', 0.11852143423034814),
     ('Category_Travel', 0.11826490901504134),
     ('Category_Retail', 0.10982816096000932),
     ('AnomalyScore', 0.09233930549618687),
     ('Category_Online', 0.0730783913737289),
     ('Category_Other', 0.07098887417973092),
     ('TransactionAmount', 0.05742035545131399),
     ('Weekday', 0.04940640276100141),
     ('Category_Food', 0.032314832357819595),
     ('SuspiciousFlag', 0.010550579883244461)]



The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.

Based on the above, the most important factors are "Amount", "OfficeHour", "AmtBalRatio", "Category_Travel" and "Category_Retail".

## 7. Conclusion

In this project, we explored the use of various machine learning models for the task of fraud detection. We trained and compared the performance of four different classification models: Logistic Regression, Decision Tree, Random Forest, and XGBoost.

The key findings from our analysis are:
- **Decision Tree Classifier Outperforms Other Models**: Among the four models evaluated, the Decision Tree Classifier demonstrated the best performance in terms of the recall score, which is a critical metric for fraud detection applications.
- **Trade-off Between Recall and other metrics**: While the Decision Tree model demonstrated the highest recall score, its precision score was relatively lower compared to the other models. This suggests a trade-off between maximizing the detection of fraudulent transactions (recall) and minimizing the number of false positives (precision). Depending on the specific requirements and priorities of the fraud detection system, the model selection and thresholds may need to be adjusted to strike the right balance between these two metrics.
- **Limited Customer Data Attributes**: The dataset used in this project had customer data that was encoded or anonymized potentially due to the reason that the data may be synthetically generated. As a result, many important features related to the customers were not included in the analysis. In a real-life fraud detection system, having access to comprehensive customer information, such as demographic details, residential location, purchase history, and behavioral patterns, could provide valuable insights and further enhance the model's predictive capabilities. 
- **Overfitting**: All models seem to performed very well on the training data, whilstthe performance metrics dropped significantly on testing data. To further improve the generalization capabilities of the model, it may be worth increasing the size and diversity of the training data. 

