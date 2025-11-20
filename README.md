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
**Visualizing the Geographical Data**

* A Scatter Plot to visualize the housing density is the best way to look at the data more closely. We can examine housing prices. Each circle represents the district's population. which is labelled as 's'. 'c' is color which represents our median_house_value. Jet is a predefined color map availabe which has 2 values blue and red representing low values and high values respectively. I have also used alpha option as 0.4 which makes it much easier to visualize the places where there has high density.

<img width="640" height="480" alt="Geographical Data" src="https://github.com/user-attachments/assets/bdfc5636-2c27-481d-ab70-f3f30540f25b" />

**Correlations**
- I have used corr method to compute the standard correlation between every pair of attributes. I also used Pandas' scatter matrix function to check the Correlations between attributes. Here is a snapshot of it.

<img width="1021" height="707" alt="image" src="https://github.com/user-attachments/assets/0a3b0fe6-175c-49f5-903f-e7a78f2c4e95" />

**Data Preparation**
- Machine Learning Algorithms cannot work with missing attributes. Previously, we saw that total bedrooms attribute has some missing values. Scikit Learn has a class which helps us fill the missing values: SimpleImputer.
I'll create an instance and replace all the missing values with median of the attribute. Median can only be calculated on the numerical arributes hence, I'll work on a copy of the data without ocean proximity.

- We only took care of the numerical attributes previously. Now, we will be looking at the text attribute and in this case, we only have one: Ocean Proximity. I'll use Scikit Learns' Ordinal Encoder Class to convert all the categorical attributes into numbers. There is an issue with how we have representated the values. ML Algorithms assumes that two nearby values are more similar than two distant values. While true in some instances ('bad', 'good', 'excellent'). That is not the case in Ocean Proximity. I will apply One Hot Encoding which provides OneHotEncoder class to convert all the Categroical values into One Hot Vectors. They are called dummy attributes.   

```javascript
#@ Preaparing the Data:
housing = strat_train_set.drop("median_house_value", axis = 1)      #Drop the lables from the Traning set.
housing_labels = strat_train_set["median_house_value"].copy()
#@ Working on the missing values:
incomplete_rows = housing[housing.isnull().any(axis=1)].head()
IPython.display.display(incomplete_rows)                          #Inspecting the missing values

incomplete_rows_dropped = incomplete_rows.dropna(subset=["total_bedrooms"], axis = 0) # Getting rid of corresponding districts

median = housing["total_bedrooms"].median()                        #Calculating the median

incomplete_rows["total_bedrooms"].fillna(median, inplace = True)  #Filling the median values

print("\nRows after filling NaNs in total_bedrooms with median:") #Inspecting the data after filling the missing values with median
IPython.display.display(incomplete_rows)

#@ Missing Values:
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")                           #Instantiating the Imputer
housing_num = housing.drop("ocean_proximity", axis=1)                  #Removing the Text Attribute
imputer.fit(housing_num)                                               #Fitting the imputer into Trainig Data
print(imputer.statistics_)

print(housing_num.median().values)                                     #Inspecting the median values
X = imputer.transform(housing_num)                                     #Transforming the Traning data set with trained Imputer
housing_imputed = pd.DataFrame(X, columns = housing_num.columns)       #Createing the DataFrame with numpy arrays
print("\n")
IPython.display.display(housing_imputed.head())

#@ Handling Text and Categroical Attribute:
from sklearn.preprocessing import OrdinalEncoder
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head())                                         #Inspecting the Categorical attribute
ordinal_encoder = OrdinalEncoder()                                #Instantiating the Encoder
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)  #Encoding the Text
IPython.display.display(housing_cat_encoded[:10])                 #Inspecting the Encoded Text
print("\n")
print(ordinal_encoder.categories_)                                #Inspecting the Categories

#@ One Hot Encoding:
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse_output = False)                #Instantiating OneHotEncoder
housing_cat_onehot = cat_encoder.fit_transform(housing_cat)       #Encoding the text
print(housing_cat_onehot[:10])
print("\n")
print(cat_encoder.categories_)                                    #Inspecting the list of categories

```

**Training the Model**
- We have to apply feature scaling in our data because ML algorithims doesn't perform well when the input numerical attributes have very different scales. We hav two common ways to get the same scale: a. Min-Max Scaling (Normalization) and b. Standarization
- Inorder to simply the many steps on Data Transformation, We will use Sciki Learn's Pipeline class to create a proper sequence. We have been handling the Categorical and Numerical Columns separetely until now. It's easier to have a single Transformer handle all the columns applying appropriate transmformations to each column. Scikit Learn has ColumnTransformer class which we will apply to all the Transformations to the Housing Data.
  - Linear Regression: I will train the Machine Learning Model using Linear Regression.

```javascript
#@ Training the Model:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()                              #Instantiating the Model
lin_reg.fit(housing_prepared, housing_labels)             #Training the Linear Model

#@ Inspecting the Model on few instances:
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_prepared = pipeline.transform(some_data)
print(f"Predictions: {lin_reg.predict(some_prepared)}")   #Inspecting the Predictions
print(f"Labels: {list(some_labels)}")

Predictions: [ 85657.90192014 305492.60737488 152056.46122456 186095.70946094
 244550.67966089]
Labels: [72100.0, 279600.0, 82700.0, 112500.0, 238300.0]

#@ Closer look at the Errors:
from sklearn.metrics import mean_squared_error, mean_absolute_error
housing_prediction = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)                                                          #Inspecting the root mean squared error

lin_mae = mean_absolute_error(housing_labels, housing_prediction)
print(lin_mae)                                                           #Inspecting the mean absolute error

68627.87390018745
49438.66860915802
```
  - Decision Trees: The obtained score is not very satisfying. It's clear that our Model is underfitting the training data. The features doesn't provide enough information to make good predictions. We can tackle this by selecting a more powerful model to feed the training algorithm with better features or by reducing the constrains on the model. I'll try a more complex model which is DecisionTreeRegressor before deciding on complicating it any further.

```javascript

#@ Implementation of Decision Trees:
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state = 11)           #Instantiating the Model
tree_reg.fit(housing_prepared, housing_labels)                #Training the Model
housing_prediction = tree_reg.predict(housing_prepared)      #Making Predictions

tree_mse = mean_squared_error(housing_labels, housing_prediction)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)                                              #Inspecting the Root Mean Squared Error

0.0

#@ Implementation of Cross Validation with Decision Trees:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#@ Inspecting the Results:
def display_scores(scores):
  print(f"Scores: {scores}")
  print(f"Mean: {scores.mean()}")
  print(f"Standard Deviation: {scores.std()}")

display_scores(tree_rmse_scores)                              #Inspecting the scores

#@ Implementation of Cross Validation with Linear Regression:
print("\n")
lin_scores= cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)                               #Inspecting the scores

Scores: [72245.94614105 69219.13480855 69507.37626747 72004.10673002 68106.4397735  77285.38495403 72048.80518577 73846.31128012 69566.5579257  70243.58659565]
Mean: 71407.36496618605
Standard Deviation: 2569.1965443766762


Scores: [71762.76364394 64114.99166359 67771.17124356 68635.19072082 66846.14089488 72528.03725385 73997.08050233 68802.33629334 66443.28836884 70139.79923956]
Mean: 69104.07998247063
Standard Deviation: 2880.3282098180694
```
  - Cross Validation: The decision tree model tells us that there are no errors.  It means that the Model has baldy overfit the Data. I'll use part of the training set for training and Model Validation of the data. Above, I've used Scikit Learn's Corss-Validation feature which splits the training set into subsets called folds. It trains and evaluates the Decision Tree model, picking a different fold for evaluation every time and training on the other remaining folds. Cross-Validation expects a utility function (greater is better) rather than a cost function (lower is better). Hence, the scoring function is opposite of MSE which is a negative value. I will use minus sign before calculation of square root for this reason. The results above shows us that Decision Tree Model is overfitting and performs worse than Linear Regression Model.

**Random Forest Regressor**
- Random Forests work by training many Decision Trees on a random subsets of the features and then averages out the predictions. I'll build a model on top of other models. This is called Ensemble Learning and this is a great way to push Machine Learning Algorithms.
  
- Grid Search and Tuning the Model: I'll use Scikit Learn's Grid Search to fine tune the data. I can tell which hyperparameters I want to experiment with and what values to try and it will evaluate all possible combinations of hyperparameters using CV. I also saved the model so that we can always access it easily. I'll save the models by using the joblib library which is more effictive at serializing large Numpy arrays. I'll save one at the very end too. 

```javascript
#@ Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators = 160, min_samples_leaf = 3, n_jobs = -1, random_state = 42)                  #Instantiating the Model
forest_reg.fit(housing_prepared, housing_labels)                                                                              #Training the Model
housing_prediction = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_prediction)                                                           #Calculating mean squared error
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)                                                                                                            #Inspecting the root mean squared error
print("\n")

27721.962704061087

Scores: [51537.44644114 48652.38768506 46540.95280581 52149.30883492 47142.82796449 51387.3530314  52089.39931894 49830.72242666 48443.56622348 53994.04142916]
Mean: 50176.80061610752
Standard Deviation: 2308.8748302001177

#@ Random Forest Regressor with Cross Validation:
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
forest_scores_rmse = np.sqrt(-forest_scores)
display_scores(forest_scores_rmse)                                                                                            #Inspecting the scores
#@ Saving the Random Forest Model:
import joblib
joblib.dump(forest_reg, "forest_reg.pkl")
                                        
['forest_reg.pkl']

from sklearn.model_selection import GridSearchCV
param_grid = [
    {"n_estimators": [30, 60, 90, 120], "max_features": [2,3,4]},
    {"bootstrap": [False], "n_estimators": [90, 120], "max_features": [2,3,4]}
]
forest_reg = RandomForestRegressor(random_state = 42)
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = "neg_mean_squared_error", return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)

#@ Inspecting the Results:
print(grid_search.best_params_)
print("\n")
print(grid_search.best_estimator_)
print("\n")

#@ Inspecting the Scores:
cvresults = grid_search.cv_results_
for mean_score, params in zip(cvresults["mean_test_score"], cvresults["params"]):
  print(np.sqrt(-mean_score), params)
print("\n")
IPython.display.display(pd.DataFrame(grid_search.cv_results_))
```
**Results**
- Several regression models were trained and evaluated using cross-validation (CV) on the training set:
  Model	                      Cross-Validation RMSE (Approx.)
  Random Forest Regressor	    ~50,176.80
  Decision Tree Regressor	    ~71,407.36
  Linear Regression	          ~69,104.08
- The Random Forest Regressor  is the lowest, confirming it has the best average performance with a mean RMSE of $50,177.

**Fine Tuning and Evaluation**
- I'll use Scikit Learn's Grid Search to fine tune the data. I can tell which hyperparameters I want to experiment with and what values to try and it will evaluate all possible combinations of hyperparameters using CV.

```javascript
from sklearn.model_selection import GridSearchCV
param_grid = [
    {"n_estimators": [30, 60, 90, 120], "max_features": [2,3,4]},
    {"bootstrap": [False], "n_estimators": [90, 120], "max_features": [2,3,4]}
]
forest_reg = RandomForestRegressor(random_state = 42)
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring = "neg_mean_squared_error", return_train_score = True)

grid_search.fit(housing_prepared, housing_labels)

#@ Inspecting the Results:
print(grid_search.best_params_)
print("\n")
print(grid_search.best_estimator_)
print("\n")

#@ Inspecting the Scores:
cvresults = grid_search.cv_results_
for mean_score, params in zip(cvresults["mean_test_score"], cvresults["params"]):
  print(np.sqrt(-mean_score), params)
print("\n")
IPython.display.display(pd.DataFrame(grid_search.cv_results_))

#@ Analyzing the Best Models and Errors:
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
cat_encoder = pipeline.named_transformers_["cat"]
cat_onehot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_onehot_attribs
sorted(zip(feature_importances, attributes), reverse = True)

#@ Evaluating the System on Testset:
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"].copy()  
#@ Preparing the Data:
X_test_prepared = pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
#@ Evaluating the Model:
final_mse = mean_squared_error(y_test, final_predictions)           #Calculating the mean squared error
final_rmse = np.sqrt(final_mse)                                     #Calculating the root mean squared error
print(final_rmse)

46043.82242621413

```
**Confidence Interval**
- Instead of just getting a single number (a point estimate) for the RMSE, this gives us a range (an interval estimate) where the true, generalized error of out model is likely to fall.

```javascript
#@ Computing a 95% confidence interval:
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc = squared_errors.mean(), scale = stats.sem(squared_errors)))

array([44117.95875455, 47892.30504544])
```
- We can be $95\%$ confident that the true error of this prediction system, if deployed, would fall within the range of $\$44,117.96$ and $\$47,892.31$. This narrow range confirms that the model is highly stable and provides a reliable estimate of California housing prices.




 


