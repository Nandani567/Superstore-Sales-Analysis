import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# LOAD DATASET

df=pd.read_csv("DATASET/cleaned_superstore.csv")
print(df.shape)
print(df.columns.to_list())

# INSIGHT QUESTIONS 
print("\nWhich city have highest sales")
top_cities_sales = (
    df.groupby(['city', 'state'])
    .agg(
        total_sales=('sales', 'sum'),
        total_orders=('order_id', 'nunique'),
        avg_order_value=('sales', 'mean')
    )
    .reset_index()
    .sort_values(by='total_sales', ascending=False)
)

print("--- TOP 10 CITIES BY HIGHEST SALES ---")
print(top_cities_sales.head(10).to_string(index=False))

yearly_product_sales = (
    df.groupby(['order_year', 'product_name'])
    .agg(total_sales=('sales', 'sum'), total_quantity=('sales', 'count'))
    .reset_index()
)

top_products_per_year = yearly_product_sales.sort_values(
    ['order_year', 'total_sales'], ascending=[True, False]
).groupby('order_year').first().reset_index()

# 3. Print the results
print("\nWhich product sale most in which year:")
print("=" * 65)
for _, row in top_products_per_year.iterrows():
    year = row['order_year']
    product = row['product_name']
    sales = row['total_sales']
    print(f"Year {year}: {product} (Total Sales: ${sales:,.2f})")

shipping_perf=(df.groupby('ship_mode').agg(avg_delay_days=('ship_delay_days','mean'),max_delay=('ship_delay_days','max'),total_orders=('order_id','nunique'))).reset_index().sort_values(by='avg_delay_days')
print("="*65)
print(f"\nShipping Performance & Delay")
print(shipping_perf.to_string(index=False))

print("\n==================================================")
print(" REVENUE BY CUSTOMER SEGMENT")
print("==================================================")
segment_perf = df.groupby('segment').agg(
    total_sales=('sales', 'sum'),
    total_orders=('order_id', 'nunique'),
    unique_customers=('customer_id', 'nunique')
).reset_index().sort_values(by='total_sales', ascending=False)

print(segment_perf.to_string(index=False))

print("=====================================================================")
print('What percentage of total revenue comes from our top 10% of customers?')
total_rev=df.groupby('customer_id').agg(
    total_revenue=('sales','sum'))
total_rev=total_rev.sort_values('total_revenue',ascending=False)
top_10=total_rev.head(int(len(total_rev)*0.10))
percentage=(top_10['total_revenue'].sum() / total_rev['total_revenue'].sum())*100
print(f"Top 10% of customers generate {percentage:.2f}% of total revenue")

print("\n===============================================================")
print(f"How many customers ordered only once versus customers who ordered more than once?")

order_frequency = df.groupby('customer_id')['order_id'].nunique()
one_time = (order_frequency == 1).sum()
returning = (order_frequency > 1).sum()

print("Customers who ordered once:", one_time)
print("Returning customers:", returning)

print("=====================================================================")
print(f"\nDo customers who buy Technology products also purchase Office Supplies in the same order?")
order_categories = df.groupby('order_id')['category'].apply(set)
both_categories=order_categories.apply(
    lambda  x: 'Technology' in x and 'Office Supplies' in x
)
technology_orders = order_categories.apply(
    lambda x: 'Technology' in x
)

both_orders = order_categories.apply(
    lambda x: 'Technology' in x and 'Office Supplies' in x
)

percentage = (both_orders.sum() / technology_orders.sum()) * 100
print(f"{percentage:.2f}% of Technology orders also contain Office Supplies")

print("\n===================================================================")
print(f"Which cities/states have lots of orders but relatively low revenue?")
print("====================================================================")
city_performance = df.groupby('city').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('sales', 'sum')
)

city_performance['revenue_per_order'] = (
    city_performance['total_revenue'] /
    city_performance['total_orders']
)

print(city_performance.sort_values('revenue_per_order'))

state_performance = df.groupby('state').agg(
    total_orders=('order_id', 'nunique'),
    total_revenue=('sales', 'sum')
)

state_performance['revenue_per_order'] = (
    state_performance['total_revenue'] /
    state_performance['total_orders']
)
print("\n States Revenue")
print(state_performance.sort_values('revenue_per_order'))



print("\n==================================================")
print("Which category makes the most money in each region?")
regional_category = df.groupby(
    ['region', 'category']
).agg(
    total_revenue=('sales', 'sum')
).reset_index()
regional_category = regional_category.sort_values(
    ['region', 'total_revenue'],
    ascending=[True, False]
)
top_category = (
    regional_category
    .groupby('region')
    .first()
    .reset_index()
)

print(top_category.groupby('region').first())

print("\n===================================================")
print("How many states make up 50% or more of total revenue?")
print("======================================================")

state_revenue=df.groupby('state')['sales'].sum()
state_revenue=state_revenue.sort_values(ascending=False)
cumulative_percentage = (
    state_revenue.cumsum() /
    state_revenue.sum()
) * 100
states_for_50 = (cumulative_percentage < 50).sum() + 1

print(cumulative_percentage.head(states_for_50))

print(f"{states_for_50} states account for at least 50% of total revenue")

print("\n===============================================")
print("Monthly Growth Rate")
df['order_date'] = pd.to_datetime(df['order_date'])
monthly_revenue = df.groupby(
    df['order_date'].dt.to_period('M')
)['sales'].sum()

mom_growth = monthly_revenue.pct_change() * 100
print(mom_growth.round(2))

print("\n=====================================================")
print("For each year, which category generated the most sales?")
yearly_category = df.groupby(
    ['order_year', 'category']
)['sales'].sum().reset_index()

yearly_category = yearly_category.sort_values(
    ['order_year', 'sales'],
    ascending=[True, False]
)

top_category_each_year = (
    yearly_category
    .groupby('order_year')
    .first()
    .reset_index()
)

print(top_category_each_year)

print("\n=========================================================")
print("Highest Average Order Volume by Month")
monthly_orders = df.groupby(
    ['order_year', 'order_month']
)['order_id'].nunique().reset_index()
average_monthly_orders = monthly_orders.groupby(
    'order_month'
)['order_id'].mean()
average_monthly_orders = average_monthly_orders.sort_values(
    ascending=False
)

print(average_monthly_orders)


# ============================================VISUALIZATION==========================================

sns.set_theme(style="darkgrid")
plt.rcParams['font.size']=10

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

subcat_sales = (
    df.groupby('sub-category')['sales']
    .sum()
    .reset_index()
    .sort_values(by='sales', ascending=False)
)

sns.barplot(
    data=subcat_sales.head(10),
    x='sales',
    y='sub-category',
    palette='Blues_r',
    ax=axes[0, 0]
)
axes[0, 0].set_title('Top 10 Sub-Categories by Total Sales', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Total Sales ($)')
axes[0, 0].set_ylabel('Sub-Category')


# ----------------------------------------------------
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

monthly_sales = df.groupby('order_month')['sales'].sum().reindex(month_order).reset_index()

sns.lineplot(
    data=monthly_sales,
    x='order_month',
    y='sales',
    marker='o',
    color='teal',
    linewidth=2.5,
    ax=axes[0, 1]
)
axes[0, 1].set_title('Aggregated Monthly Sales (Seasonality)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Total Sales ($)')
axes[0, 1].tick_params(axis='x', rotation=45)

# ----------------------------------------------------
# Plot 3: Shipping Delays by Ship Mode (Boxplot / Bar)
# ----------------------------------------------------
sns.barplot(
    data=df,
    x='ship_mode',
    y='ship_delay_days',
    palette='Oranges_r',
    ax=axes[1, 0]
)
axes[1, 0].set_title('Average Shipping Delay (Days) per Ship Mode', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Shipping Mode')
axes[1, 0].set_ylabel('Average Delay (Days)')



segment_sales = df.groupby('segment')['sales'].sum().reset_index()

sns.barplot(
    data=segment_sales,
    x='segment',
    y='sales',
    palette='Greens_r',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Total Sales Revenue by Customer Segment', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Customer Segment')
axes[1, 1].set_ylabel('Total Sales ($)')

plt.tight_layout()

plt.show()

