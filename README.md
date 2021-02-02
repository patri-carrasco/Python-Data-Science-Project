# Data junior course


## Iris
As this exercise I have built an automatic learning model with the scikit-learn package in Python to build a simple classification model (to classify Iris flowers) using the random forest algorithm


I have followed the video of the Professor Data Professor https://youtu.be/XmSlFPDjKdc.



## Comparing-classifiers

We will generate a synthetic classification dataset and compare an exhaustive set of 14 machine learning algorithms from the scikit-learn package.

####  Build Classification Models
### 1. Defining learning classifiers
~~~names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
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
~~~
scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    scores.append(score)
~~~

