
f"""

###########################################################################
CLTV CALCULATE
###########################################################################

CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
Customer_Value = Average_Order_Value * Purchase_Frequency
Average_Order_Value = Total_Revenue / Total_Number_of_Orders
Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
Churn_Rate = 1 - Repeat_Rate
Profit_margin

"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_yedek = pd.read_excel("datasets/online_retail_II.xlsx",
                         sheet_name="Year 2010-2011")

df = df_yedek.copy()
df.head()


# Data Preparation

df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby("Customer ID").agg({"Invoice": lambda x: len(x),
                                         "Quantity": lambda x: x.sum(),
                                         "TotalPrice": lambda x: x.sum()})

cltv_df.columns = ["total_transaction", "total_unit", "total_price"]

# There are 0 total_price values available.
cltv_df = cltv_df[(cltv_df["total_price"]) > 0]

# total_transaction: number of transactions
# total_unit: How many units she bought in transactions
# total_price: Income provided

# 1. Calculate Average Order Value

cltv_df["average_order_value"] = cltv_df["total_price"] / cltv_df["total_transaction"]

# 2. Calculate Purchase Frequency

df["Customer ID"].nunique()

cltv_df["purchase_frequency"] = cltv_df["total_transaction"] / cltv_df.shape[0]

# 3. Calculate Repeat Rate and Churn Rate
# Repeat Rate: Number of customers who shop at least once / number of all customers


repeat_rate = cltv_df[cltv_df["total_transaction"] > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate

# 4. Calculate Profit Margin

cltv_df["profit_margin"] = cltv_df["total_price"] * 0.05

# 5. Calculate Customer Lifetime Value

cltv_df["CV"] = (cltv_df["average_order_value"] * cltv_df["purchase_frequency"]) / churn_rate

cltv_df["CLTV"] = cltv_df["CV"] * cltv_df["profit_margin"]

cltv_df.sort_values("CLTV", ascending=False)

# Let's apply transformations for easy expression

scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])
cltv_df.sort_values("CLTV", ascending=False)
