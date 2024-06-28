# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

pd.options.mode.chained_assignment = None
task = pd.read_csv(r"C:\Users\User\Downloads\retailer_data.csv")

# task 1
# getting rid of null and n/a values
columns_needed = ['sale_price', 'units_sold', 'rating', 'page_views']
data_to_clean = task[columns_needed]
data_to_clean = data_to_clean.dropna()
task.update(data_to_clean)
alpha = task
task['revenue'] = alpha['sale_price'] * alpha['units_sold']


# adding dollar sign to all sales values
data_to_clean['sale_price'] = data_to_clean['sale_price'].apply(lambda x: f'${x}')

# checking if all rating are between range 1 and 5
not_between_1_and_5 = ~data_to_clean['rating'].between(1, 5)

# Counting the number of values not between 1 and 5
count_not_between_1_and_5 = not_between_1_and_5.sum()


# checking to see if theirs any views that are too high in page_views
print(data_to_clean['units_sold'].mean())
print(data_to_clean['units_sold'].max())
print(data_to_clean['units_sold'].min())


Q1 = data_to_clean['units_sold'].quantile(0.25)
Q3 = data_to_clean['units_sold'].quantile(0.75)
iqr = Q3 - Q1
print(f"Q1: {Q1}", f"Q3: {Q3}", f"iqr:{iqr}")

# checking to see if theirs any values that are too high in 'page_views'
print(data_to_clean['page_views'].mean())
print(data_to_clean['page_views'].max())
print(data_to_clean['page_views'].min())

# count how many values are equal to maximum and minim

equal_to_specific_value0 = data_to_clean['page_views'] == data_to_clean['page_views'].max()
equal_to_specific_value_20 = data_to_clean['page_views'] == data_to_clean['page_views'].min()
# Counting the number of values equal to the specific value
count_equal_to_specific_value0 = equal_to_specific_value0.sum()
count_equal_to_specific_value_20 = equal_to_specific_value_20.sum()
print(count_equal_to_specific_value0)
print(count_equal_to_specific_value_20)
Q1 = data_to_clean['page_views'].quantile(0.25)
Q3 = data_to_clean['page_views'].quantile(0.75)
iqr = Q3 - Q1
print(f"Q1: {Q1}", f"Q3: {Q3}", f"iqr:{iqr}")

# q2
# add revenue tab
task['revenue'] = alpha['sale_price'] * alpha['units_sold']
# subsets by unit sold
specific_ans = task[task['units_sold'] > 10000]
Q3_rating = specific_ans['rating'].quantile(0.75)
Q3_revenue = specific_ans['revenue'].quantile(0.75)
specific = ['revenue', 'rating']
even_more_specified = specific_ans[specific]


def categorize(row):
    if row['revenue'] > 240000 and row['rating'] > 4.09:
        return 'Above 240000, Rating Above 4.09'
    elif row['revenue'] <= 240000 and row['rating'] > 4.09:
        return 'Below 240000, Rating Above 4.09'
    elif row['revenue'] > 240000 and row['rating'] <= 4.09:
        return 'Above 240000, Rating Below 4.09'
    else:
        return 'Below 240000, Rating Below 4.09'


even_more_specified['Category'] = even_more_specified.apply(categorize, axis=1)

# Count the occurrences of each category
category_counts = even_more_specified['Category'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Revenue and Rating Categories for 10,000 units sold')
plt.show()

# task 3

category = ['page_views', 'uses_ad_boosts']
task.update(data_to_clean)
ad_quantity = task[category]

# subset data based on use of ad-boost
subset_ad0 = ad_quantity[ad_quantity['uses_ad_boosts'] == 0]
subset_ad1 = ad_quantity[ad_quantity['uses_ad_boosts'] == 1]

print(f"mean units doesn't use ad boost: {subset_ad0.mean()}",
      f"max and min doesn't use ad :{max(subset_ad0['page_views']), min(subset_ad0['page_views'])}",
      f"mean units does use ad boost: {subset_ad1.mean()}",
      f"max and min uses ad :{max(subset_ad1['page_views']), min(subset_ad1['page_views'])}"
      f"variance doesn't use ad and does use ad: {subset_ad0['page_views'].var(), subset_ad1['page_views'].var()}")

category = ['units_sold', 'uses_ad_boosts']
ad_quantity = task[category]

# Create a Boolean mask for the condition
condition_adboost = ad_quantity['uses_ad_boosts'] == 1
condition_nonadboost = ad_quantity['uses_ad_boosts'] == 0
# Use the sum() method to count the number of True values
count1 = condition_adboost.sum()
count2 = condition_nonadboost.sum()

# subset data based on use of adboost
subset_ad0 = ad_quantity[ad_quantity['uses_ad_boosts'] == 0]
subset_ad1 = ad_quantity[ad_quantity['uses_ad_boosts'] == 1]

ad1_popular = subset_ad1['units_sold'] >= 10000
count_Ad1_popular = ad1_popular.sum()

ad0_popular = subset_ad0['units_sold'] >= 10000
count_ado_popular = ad0_popular.sum()

print(f"mean units doesn't use ad boost: {subset_ad0.mean()}", f"mean units does use ad boost: {subset_ad1.mean()}",
      f"percentage uses ad is popular:{count_Ad1_popular * 100 / len(ad1_popular)}",
      f"percentage doesnt use ad is popular:{count_ado_popular * 100 / len(ad0_popular)}")
      


# Task 4 the classification model binary classification popular not popular

specific = ['units_sold', 'uses_ad_boosts', 'rating', 'revenue']
classification_data_set = task[specific]

classification_data_set['RevenueBelow240000'] = classification_data_set['revenue'] < 240000
classification_data_set['RatingBelow4.09'] = classification_data_set['rating'] < 4.09

#  if UnitsSold is above 10000 give it 1 if not give it a zero
classification_data_set['UnitsSoldAbove10000'] = (classification_data_set['units_sold'] > 10000).astype(int)

# Select features and response variable
# X = classification_data_set[['RevenueBelow240000', 'RatingBelow4.09', 'add_boosted']]
X = classification_data_set[['RevenueBelow240000', 'RatingBelow4.09', 'uses_ad_boosts']]
y = classification_data_set['UnitsSoldAbove10000']

# break model into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# intialize classifier
clf = RandomForestClassifier(random_state=42)

# train classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
