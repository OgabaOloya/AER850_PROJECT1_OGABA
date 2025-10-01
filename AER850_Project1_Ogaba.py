# ================================
#Ogaba Oloya
#AER 850 Project 1
#501097689

#------------------------------------------------------------------------------
#Step 1: Data Processing 

#Importing our relevant toolkits into the system for usage later
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Reading the data from the file 
data = pd.read_csv("AER850_Project1Data/Project 1 Data.csv")

#Checking to verify that all of the columns are valid and to further assign manually by reading without headers
if any (col.lower().startswith("unnamed")for col in data.columns):
    data = pd.read_csv(
        "AER850_Project1Data/Project 1 Data.csv", 
        header=None, 
        names=["X", "Y", "Z", "Step"])

#Displaying the first few rows as a quick verification check 
print(data.head())


#------------------------------------------------------------------------------
#Step 2: Data Visualization 
print("\n\n-----------------Step 2: Data Visualization-----------------\n\n")

#Setting up boxplot subplots for X, Y, and Z
plt.figure(figsize=(15,5))

#Box plot for the variable X
plt.subplot(1,3,1)
plt.hist(data["X"], bins=20, color="lightgreen", edgecolor="black")
plt.title("Histogram of X")
plt.xlabel("X")
plt.ylabel("Frequency")

#Box plot for the variable Y
plt.subplot(1,3,2)
plt.hist(data["Y"], bins=20, color="lightblue", edgecolor="black")
plt.title("Histogram of Y")
plt.xlabel("Y")
plt.ylabel("Frequency")

#Box plot for the variable Z
plt.subplot(1,3,3)
plt.hist(data["Z"], bins=20, color="salmon", edgecolor="black")
plt.title("Histogram of Z")
plt.xlabel("Z")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
#Step 3: Correlation Analysis 
print("\n\n-----------------Step 3: Correlation Analysis-----------------\n\n")

# Compute our correlations
corr = data.corr(method='pearson', numeric_only=True)

#Plotting the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(data=corr, annot=True)
plt.title("Pearson Correlation Heatmap")
plt.show()

#Running an 80% training sample to learn patterns in Z per lecture material 
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

X_train = train_data[["X", "Y", "Z"]]
y_train = train_data["Step"]
X_test = test_data[["X", "Y", "Z"]]
y_test = test_data["Step"]

# Compute correlation
for col in ["X", "Y", "Z"]:
    corr_value = data[[col, "Step"]].corr().loc["Step", col]
    print(f"{col} vs Step: {corr_value:.3f}")

#------------------------------------------------------------------------------
#Step 4: Classification Model Development/Engineering  
print("\n\n-----------------Step 4: Classification Model Development/Engineering-----------------\n\n")

#Importing our slected classifiers from scikit-learn's module
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Importing our three model evaluation models 
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

#-------------------------------------
#Part A: Logistic Regression (Baseline Model --> GridSearch Variation)
#-------------------------------------

print("\n--- Logistic Regression ---")

Regression = LogisticRegression(max_iter=1000)
Regression.fit(X_train, y_train)

print("Training Data Accuracy:", round(Regression.score(X_train, y_train), 3))
print("New/Unseen Test Data Accuracy:", round(Regression.score(X_test, y_test), 3))

# Cross-validation using data 
cv_scores_regression = cross_val_score(Regression, X_train, y_train, cv=5, scoring="accuracy")
print("Mean CV Accuracy:", round(cv_scores_regression.mean(), 3))

#GridSearch Logistic Regression
param_grid_reg = {'C': [0.1, 1, 10],'solver': ['lbfgs', 'liblinear'],'penalty': ['l2']}

grid_regression = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid_reg,
    scoring='accuracy',
    cv=5
)
grid_regression.fit(X_train, y_train)

print("Best Regression Hyperparameters found by GridSearch:", grid_regression.best_params_)
print("Best Tuned Model Regression Accurary:", round(grid_regression.best_score_, 3))

#-------------------------------------
#Part B:Descision Tree (GridSearch Variation)
#-------------------------------------

print("\n--- Decision Tree ---")

DecisionTree = DecisionTreeClassifier(random_state=42)
DecisionTree.fit(X_train, y_train)  

print("Training Data Accuracy:", round(DecisionTree.score(X_train, y_train), 3))
print("New/Unseen Test Data Accuracy:", round(DecisionTree.score(X_test, y_test), 3))

# Cross-validation using data
cv_scores_decisiontree = cross_val_score(DecisionTree, X_train, y_train, cv=5, scoring="accuracy")
print("Mean CV Accuracy:", round(cv_scores_decisiontree.mean(), 3))

# GridSearch Decision Tree
hyperparam_gridsearch_descisiontree = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_decisiontree = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=hyperparam_gridsearch_descisiontree,
    scoring="accuracy",
    cv=5
)
grid_decisiontree.fit(X_train, y_train)

print("Best Decision Tree Hyperparameters found by GridSearch:", grid_decisiontree.best_params_)
print("Best Tuned Model Decision Tree Accuracy:", round(grid_decisiontree.best_score_, 3))

#-------------------------------------
#Part C:Support Vector Machine (GridSearch Variation)
#-------------------------------------

print("\n--- Support Vector Machine (SVM) ---")

SVM = SVC(random_state=42)
SVM.fit(X_train, y_train)

print("Training Data Accuracy:", round(SVM.score(X_train, y_train), 3))
print("New/Unseen Test Data Accuracy:", round(SVM.score(X_test, y_test), 3))

# Cross-validation using data
cv_scores_svm = cross_val_score(SVM, X_train, y_train, cv=5, scoring="accuracy")
print("Mean CV Accuracy:", round(cv_scores_svm.mean(), 3))

# GridSearch SVM
hyperparam_gridsearch_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=hyperparam_gridsearch_svm,
    scoring="accuracy",
    cv=5
)
grid_svm.fit(X_train, y_train)

print("Best SVM Hyperparameters found by GridSearch:", grid_svm.best_params_)
print("Best Tuned Model SVM Accuracy:", round(grid_svm.best_score_, 3))

#-------------------------------------
#Part D:Support Vector Machine (RandomizedSearch Variation)
#-------------------------------------
print("\n--- Randomized Search Support Vector Machine (SVM) ---")

# Randomized Search for SVM
hyperparam_distribution_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

randomsearch_svm = RandomizedSearchCV(SVC(), param_distributions=hyperparam_distribution_svm, n_iter=10, cv=5, scoring="accuracy", random_state=42)
randomsearch_svm.fit(X_train, y_train)
print("Best Hyperparameters found by RandomizedSearch (RandomizedSearchCV):", randomsearch_svm.best_params_)
print("Best Tuned Model SVM (randomized) Accuracy:", round(randomsearch_svm.best_score_, 3))











