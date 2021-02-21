# Python Data Science Project


## Iris
As this exercise I have built an automatic learning model with the scikit-learn package in Python to build a simple classification model (to classify Iris flowers) using the random forest algorithm


I have followed the video of the Professor Data Professor https://youtu.be/XmSlFPDjKdc.



## Comparing-classifiers

We will generate a synthetic classification dataset and compare an exhaustive set of 14 machine learning algorithms from the scikit-learn package.

####  Build Classification Models
### 1. Defining learning classifiers
~~~ python
names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
         "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA", "SGD"]
​
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="l2")]
~~~
#### 2 Build Model, Apply Model on Test Data & Record Accuracy Scores
We create the score array to save the 14 predictions of the classifiers
~~~ python
scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    scores.append(score)
~~~


## Hyperparameter Tuning of Machine Learning Model in Python
You how to tune the hyperparameters of machine learning model in Python using the scikit-learn package.

~~~ python
max_features_range = np.arange(1,6,1)
n_estimators_range = np.arange(10,210,10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

rf = RandomForestClassifier()

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
~~~


# Machine Learning in Python: Building a Linear Regression Model
We how to build a linear regression model in Python using the scikit-learn package. We will be using the Diabetes dataset (built-in data from scikit-learn) and the Boston Housing (download from GitHub) dataset.

And We created our predictions:
* 1 predictor, RM
~~~ python

X=df[["rm"]]
# target
Y=df["medv"]
~~~
* 2 predictor, RM
~~~ python

X=df[["rm", "lstat"]]
# target
Y=df["medv"]
~~~
* all predictors but the MEDV
~~~ python

X=df.drop("medv", axis=1)
# target
Y=df["medv"]
~~~

The best prediction, medv & rn:
~~~ python
score1': 0.48352545599133423,
 'mean_squared_error1': 43.60055177116956,
 ~~~


# Machine Learning in Python: Performing Principal Component Analysis (PCA)


You how to perform principal component analysis (PCA) in Python using the scikit-learn package. PCA represents a powerful learning approach that enables the analysis of high-dimensional data as well as reveal the contribution of descriptors in governing the distribution of data clusters. Particularly, we will be creating PCA scree plot, scores plot and loadings plot.


# How to Plot an ROC Curve in Python | Machine Learning in Python

You how to plot the Receiver Operating Characteristic (ROC) curve in Python using the scikit-learn package. I will also you how to calculate the area under an ROC (AUROC) curve. In the tutorial, we will be comparing 2 classifiers via the ROC curve and the AUROC values.

# Easy Web Scraping in Python using Pandas for Data Science

You how to easily web scrape data from websites in Python using the pandas library. Particularly, the read_html() function or practically pd.read_html() will be used to extract table data of National Basketball Association (NBA) player stats from https://www.basketball-reference.com/​. We will then do some cleanup to produce the final data in the form of a dataframe. Finally, we will be doing a quick exploratory data analysis by making histogram plots.


# Cheminformatics in Python: Predicting Solubility of Molecules | End-to-End Data Science Project

Version python 3.7 

In this video, I will show you step-by-step in this End-to-end Bioinformatics / Cheminformatics tutorial on how to use Data Science in a Computational Drug Discovery project as we reproduce the research work of Delaney by predicting the solubility of molecules in Python using scikit-learn, rdkit and pandas libraries. 

### Linear Regression Model 
1º Import
~~~ python
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
LinearRegression()
~~~

2 º Predicts the X_train
~~~ python
Y_pred_train = model.predict(X_train)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_train, Y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_train, Y_pred_train))
Coefficients: [-0.75311929 -0.00647419 -0.00720234 -0.43018103]
Intercept: 0.27324370551875043
Mean squared error (MSE): 0.99
Coefficient of determination (R^2): 0.78
~~~
3º Predicts the X_test
~~~ python
Y_pred_test = model.predict(X_test)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred_test))
Coefficients: [-0.75311929 -0.00647419 -0.00720234 -0.43018103]
Intercept: 0.27324370551875043
Mean squared error (MSE): 1.08
Coefficient of determination (R^2): 0.71
~~~ 