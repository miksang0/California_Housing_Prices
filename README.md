# **Model for California Housing Prices**
**Objective**
- This project is a Machine Learning regression task focused on predicting the median house value for districts in California. The objective is to build a robust model of housing prices using data from the California Census.

**Dataset**
- The project uses the California Housing Prices Dataset, which is based on data from the 1990 California Census and obtained from the StatLib Repository. I simply downloaded the dataset from Kaggle and used it. Each row in the dataset represents a single district in California. Some metrics from the dataset are Population, Longitude, Latitude, Housing Median Age, etc. 

**Dependencies**
- The following Python libraries are required to run the notebook:
```javascript
import sys, os, tarfile, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
```
**Project Workflow**
- The project follows a standard Machine Learning pipeline, including Data Acquisition, Exploratory Data Analysis (EDA), Data Preprocessing, Model Training, Fine-Tuning, and Evaluation.

**Exploratory Data Analysis**
- Each row in the dataset represents a single district in California. The data contains 20,640 entries and 10 attributes. From those 10 attributes, 9 are numerical and 1 of them is categorical. Here is a glimpse of it. I have used the info method to take a glance at the dataset.

<img width="1500" height="1000" alt="Histogram" src="https://github.com/user-attachments/assets/12fff16f-1712-4aa3-837a-266725ecc9ca" />

- The analysis shows us that the DataFrame has 20640 non-null values except for total_bedrooms. It means values for 207 districts are missing. Also, the ocean_proximity has the datatype of an object, which means this attribute is repetitive and is most likely to be a categorical attribute.

**Stratified Sampling**
- I have used stratified sampling, which will give us proportional representation of each income category, which means less bias. I have also used Scatter Matrix and Attribute combinations using Python.
  
```javascript
#@ Stratified Sampling on the basis of Income Category:
from sklearn.model_selection import StratifiedShuffleSplit
import IPython.display

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

#@ Calculating proportions for Stratified Split
strat_props = strat_test_set["income_cat"].value_counts() / len(strat_test_set)                 #Income Category in Test Set

#@ Calculate proportions for the Overall Dataset
overall_props = housing["income_cat"].value_counts() / len(housing)

#@ Generate a Random Split for comparison purposes
train_set_rand, test_set_rand = train_test_split(housing, test_size=0.2, random_state=11)       #Splitting the dataset
random_props = test_set_rand["income_cat"].value_counts() / len(test_set_rand)

#@ Create the Comparison DataFrame
compare_props = pd.DataFrame({
    "Overall": overall_props,
    "Stratified": strat_props,
    "Random": random_props
}).sort_index()

#@ # Calculate % Error
compare_props["Random % Error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Stratified % Error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

#@ Display the results
print("Comparison of Income Category Proportions:")
IPython.display.display(compare_props)
print("\n")
```

