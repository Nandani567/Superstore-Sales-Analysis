import pandas as pd;

# Load Dataset
data=pd.read_csv("DATASET/train.csv")
# Basic cleaning
print(f"Columns: {data.columns.to_list()}")
print(f"Shape: {data.shape}")
print("\n--- Dataset Overview ---")
data.info()
print(f"\nMissing values:\n{data.isnull().sum()}")
print(f"\nDuplicates: {data.duplicated().sum()}")
print(f"\nData types:\n{data.dtypes}")
print(f"\nDescription:\n{data.describe()}")

# View rows where Postal Code is missing
missing_pc = data[data['Postal Code'].isnull()][['State', 'City', 'Postal Code']]
print(missing_pc)

# Fill Missing postal code 
data.loc[(data['City']=='Burlington')&(data['State']=='Vermont'),'Postal Code']=5401
data['Postal Code'] = data['Postal Code'].astype(int).astype(str).str.zfill(5)
print(f"Remaining nulls in Postal Code: {data['Postal Code'].isnull().sum()}")

# Recheck missing value 
check_pc = data[data['Postal Code'].isnull()][['State', 'City', 'Postal Code']]
print(check_pc)

# Convert Datetime format
data['Order Date']=pd.to_datetime(data['Order Date'], dayfirst=True)
data['Ship Date']=pd.to_datetime(data['Ship Date'], dayfirst=True)

print("--- Data Types ---")
print(data[['Order Date', 'Ship Date']].dtypes)

print("\n--- Preview First 5 Rows ---")
print(data[['Order Date', 'Ship Date']].head())

# standard columns name
data.columns = data.columns.str.lower().str.replace(' ', '_')
print("Data columns:" ,data.columns)


# drop row_id 
if 'row_id' in data.columns:
    data.drop(columns=['row_id'], inplace=True)

data['ship_delay_days'] = (data['ship_date'] - data['order_date']).dt.days
data['order_year'] = data['order_date'].dt.year
data['order_month'] = data['order_date'].dt.month_name()

categorical_cols = ['ship_mode', 'segment', 'region', 'category', 'sub-category']

for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype('category')

data.to_csv("DATASET/cleaned_superstore.csv", index=False)
print("Dataset successfully cleaned and saved as 'cleaned_superstore.csv'!")