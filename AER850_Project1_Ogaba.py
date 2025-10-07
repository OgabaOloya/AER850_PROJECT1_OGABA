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

#Histogram for the variable X
plt.subplot(1,3,1)
plt.hist(data["X"], bins=20, color="lightgreen", edgecolor="black")
plt.title("Histogram of X")
plt.xlabel("X")
plt.ylabel("Frequency")

#Histogram for the variable Y
plt.subplot(1,3,2)
plt.hist(data["Y"], bins=20, color="lightblue", edgecolor="black")
plt.title("Histogram of Y")
plt.xlabel("Y")
plt.ylabel("Frequency")

#Histogram for the variable Z
plt.subplot(1,3,3)
plt.hist(data["Z"], bins=20, color="salmon", edgecolor="black")
plt.title("Histogram of Z")
plt.xlabel("Z")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#3D Space representation of our data 
fig = plt.figure(figsize=(10, 7))
axes = fig.add_subplot(111, projection='3d')

scatter = axes.scatter(
    data['X'], data['Y'], data['Z'],
    c=data['Step'], cmap='tab20'
)

axes.set_xlabel("X")
axes.set_ylabel("Y")
axes.set_zlabel("Z")
axes.set_title("3D Visualization of X, Y, Z (Sorted by Coloured Steps)")

legend = axes.legend(*scatter.legend_elements(), title="Step", bbox_to_anchor = (1.15, 1), loc = 'upper left')

plt.show()


# Computing the mean, std, min, max of X, Y, Z grouped by Step
summary_table = data.groupby("Step")[["X", "Y", "Z"]].agg(["mean", "std", "min", "max"])

# Creating our figure for Summary Table 
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")  # Hide axes

# Convert DataFrame to table for matplotlib display
table = ax.table(
    cellText=np.round(summary_table.values, 2),  # Round to 2 decimals
    rowLabels=summary_table.index,
    colLabels=["X_mean", "X_std", "X_min", "X_max", 
               "Y_mean", "Y_std", "Y_min", "Y_max", 
               "Z_mean", "Z_std", "Z_min", "Z_max"],
    cellLoc="center",
    loc="center"
)

# Adjusting our fonts & sizing for wide tables
table.auto_set_font_size(False)
table.set_fontsize(9)  
table.scale(1.1, 1.2)  


for key, cell in table.get_celld().items():
    cell.set_edgecolor("black")

plt.title("Summary Statistics by Step", fontsize=14)
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score


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

# Confusion Matrix & Classification Report
best_regression = grid_regression.best_estimator_
y_pred_regression = best_regression.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_regression, zero_division=0))

cm_regression = confusion_matrix(y_test, y_pred_regression)
sns.heatmap(cm_regression, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Value"); plt.ylabel("Actual Value")
plt.show()

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

# Confusion Matrix & Classification Report
best_decisiontree = grid_decisiontree.best_estimator_
y_pred_decisiontree = best_decisiontree.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_decisiontree, zero_division=0))

cm_decisiontree = confusion_matrix(y_test, y_pred_decisiontree)
sns.heatmap(cm_decisiontree, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Value"); plt.ylabel("Actual Value")
plt.show()

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

#Confusion Matrix and Classification Report: 
best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Value"); plt.ylabel("Actual Value")
plt.show()

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

#Confusion Matrix and Classification Report: 
best_svm_rand = randomsearch_svm.best_estimator_
y_pred_svm_rand = best_svm_rand.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_svm_rand, zero_division=0))

cm_svm_rand = confusion_matrix(y_test, y_pred_svm_rand)
sns.heatmap(cm_svm_rand, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix - SVM (RandomizedSearch)")
plt.xlabel("Predicted Value"); plt.ylabel("Actual Value")
plt.show()

#-------------------------------------
#Part E: Threshold Analysis of all Models  
#-------------------------------------
print("\n--- Part E: Threshold Analysis of all Models ---")


# Choose one positive class for one-vs-rest thresholding
positiveclass = 1  

# Binarize ground truth labels (is class = POS_CLASS?)
y_test_bin = (y_test == positiveclass).astype(int)

# Collect all of our tuned models
best_regression   = grid_regression.best_estimator_
best_decisiontree = grid_decisiontree.best_estimator_
best_svm          = grid_svm.best_estimator_
best_rand_svm = randomsearch_svm.best_estimator_
best_rand_svm = randomsearch_svm.best_estimator_


models_for_threshold = [
    ("Best Logistic Regression", best_regression),
    ("Best Decision Tree", best_decisiontree),
    ("Best SVM GridSearch", best_svm),
    ("Best SVM RandomizedSearch", best_rand_svm),
]

# Definig our thresholds to sweep from 0.0 → 0.9 in increments of 0.1 each time 
thresholds = np.arange(0.0, 1.0, 0.1)

# Creation of a dictionary to summarize best thresholds for each model
best_thresholds_summary = {}

# Performing of threshold analysis for each tuned model
for name, model in models_for_threshold:
    
    # Getting the probability scores for positiveclass
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(positiveclass)
        y_scores = model.predict_proba(X_test)[:, class_index]
    else:
        # Fallback for models without predict_proba (like SVM with no probas enabled)
        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, cv="prefit")
        calibrated_model.fit(X_train, y_train)
        class_index = list(calibrated_model.classes_).index(positiveclass)
        y_scores = calibrated_model.predict_proba(X_test)[:, class_index]

    precisions, recalls, f1s = [], [], []
    print(f"\n{name}: Precision/Recall/F1 across thresholds (positive class = {positiveclass})")

    # Threshold sweep from 0-0.9
    for t in thresholds:
        y_pred_thr = (y_scores >= t).astype(int)
        p = precision_score(y_test_bin, y_pred_thr, zero_division=0)
        r = recall_score(y_test_bin, y_pred_thr, zero_division=0)
        f = f1_score(y_test_bin, y_pred_thr, zero_division=0)
        precisions.append(p); recalls.append(r); f1s.append(f)
        print(f"Threshold: {t:.1f}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

    # Saving the best threshold by F1
    best_idx = int(np.argmax(f1s))
    best_thresholds_summary[name] = {
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precisions[best_idx]),
        "best_recall": float(recalls[best_idx]),
        "best_f1": float(f1s[best_idx])
    }

    # Plotting Prescision/Recall/F1 vs Threshold
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.plot(thresholds, recalls, marker="o", label="Recall")
    plt.plot(thresholds, f1s, marker="o", label="F1")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.title(f"{name} - P/R/F1 vs Threshold (Class {positiveclass})")
    plt.legend(); plt.grid(True); plt.show()

# Final Summary
print("\nBest thresholds by model (max F1, one-vs-rest on Step =", positiveclass, "):")
for k, v in best_thresholds_summary.items():
    print(f"{k}: t* = {v['best_threshold']:.1f}  "
          f"(Prescision = {v['best_precision']:.3f}, Recall = {v['best_recall']:.3f}, F1 = {v['best_f1']:.3f})")
    

#-------------------------------------
# Part F: Plotting of ROC curves for all models (One-vs-Rest)
#-------------------------------------
print("\n--- Part F: Plot of ROC curves for all Models ---")

# Choose the positive class for one-vs-rest ROC
positiveclass = 1
y_test_bin = (y_test == positiveclass).astype(int)

# Models to evaluate
models = [
    ("Logistic Regression", best_regression),
    ("Decision Tree", best_decisiontree),
    ("SVM - GridSearch", best_svm),
    ("SVM - RandomizedSearch", best_rand_svm)
]

plt.figure(figsize=(8,6))

for name, model in models:
    # Get probabilities for the positive class
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(positiveclass)
        y_scores_model = model.predict_proba(X_test)[:, class_index]
    else:
        # Handle SVM models without predict_proba by calibration
        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(model, cv="prefit")
        calibrated_model.fit(X_train, y_train)
        class_index = list(calibrated_model.classes_).index(positiveclass)
        y_scores_model = calibrated_model.predict_proba(X_test)[:, class_index]

    # Compute ROC curve + AUC
    fpr, tpr, _ = roc_curve(y_test_bin, y_scores_model)
    auc_score = roc_auc_score(y_test_bin, y_scores_model)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

# Add diagonal line for reference
plt.plot([0,1], [0,1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves (One-vs-Rest, Positive Class = {positiveclass})")
plt.legend()
plt.grid(True)
plt.show()


#------------------------------------------------------------------------------
#Step 5: Model Performance Analysis 
print("\n\n-----------------Step 5: Model Performance Analysis-----------------\n\n")

from sklearn.metrics import confusion_matrix, classification_report

#-------------------------------------
#Part A: Best Estimator of Logistic Regression (GridSearch Variation)
#-------------------------------------
print("\n================ Logistic Regression Results ================\n")
best_regression = grid_regression.best_estimator_
y_pred_reg = best_regression.predict(X_test)

#Creation of 
print("Classification Report:\n", classification_report(y_test, y_pred_reg, zero_division=0))
cm_reg = confusion_matrix(y_test, y_pred_reg)
sns.heatmap(cm_reg, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Best Params)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#-------------------------------------
#Part B: Best Estimator of Descision Tree (GridSearch Variation)
#-------------------------------------
print("\n================ Decision Tree Results ================\n")
best_decisiontree = grid_decisiontree.best_estimator_
y_pred_dt = best_decisiontree.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_dt, zero_division=0))
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Decision Tree (Best Params)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#-------------------------------------
#Part C Best Estimator of SVM (GridSearch Variation)
#-------------------------------------
print("\n================ SVM (GridSearchCV) Results ================\n")

#Best tuned SVM for making predictions 
best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=0))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - SVM (Best Params)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#-------------------------------------
#Part D Best Estimator of SVM (RandomSearch Variation)
#-------------------------------------
print("\n================ SVM (RandomizedSearchCV) Results ================\n")
best_svm_rand = randomsearch_svm.best_estimator_
y_pred_svm_rand = best_svm_rand.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred_svm_rand, zero_division=0))
cm_svm_rand = confusion_matrix(y_test, y_pred_svm_rand)
sns.heatmap(cm_svm_rand, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix - SVM (RandomizedSearch -- Best Params)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#------------------------------------------------------------------------------
#Step 6: Stacked Model Performance Analysis 
print("\n\n-----------------Step 6: Stacked Model Performance Analysis-----------------\n\n")

#Importing our requiredpackage and metrics 
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Defining our base learners (which in this case are the best tuned models from Step 4)
estimators = [('logreg', best_regression),('dtree', best_decisiontree)]

# Uisng our best tuned SVM as our final estimator 
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=best_svm,
    cv=5,
    stack_method="auto",
    passthrough=False
)

# Training our stacked model and making predictions 
stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_test)

# Defining our evaluation metrics
stack_accuracy = accuracy_score(y_test, y_pred_stack)
stack_precision = precision_score(y_test, y_pred_stack, average="weighted", zero_division=0)
stack_recall = recall_score(y_test, y_pred_stack, average="weighted", zero_division=0)
stack_f1 = f1_score(y_test, y_pred_stack, average="weighted", zero_division=0)

print("\n================ Stacked Model Results ================\n")
print("Accuracy:", round(stack_accuracy, 3))
print("Precision:", round(stack_precision, 3))
print("Recall:", round(stack_recall, 3))
print("F1 Score:", round(stack_f1, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred_stack, zero_division=0))

# Confusion Matrix Visualization
cm_stack = confusion_matrix(y_test, y_pred_stack)
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Reds")
plt.title("Confusion Matrix - Stacked Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#------------------------------------------------------------------------------
#Step 7: Model Evaluation
print("\n\n-----------------Step 7: Model Evaluation-----------------\n\n")

#Importing our Joblib toolkit
import joblib 

# Saving the selected best model 
final_model = best_regression  
joblib.dump(final_model, "best_model.pkl")
print("\nModel saved as best_model.pkl")

loaded_model = joblib.load("best_model.pkl")

# Define the test points as full (X, Y, Z) coordinates
test_points = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0,   3.0625, 1.93],   
    [9.4,   3.0,    1.8],
    [9.4,   3.0,    1.3]
], columns=["X", "Y", "Z"])

# Making our predictions
predictions = loaded_model.predict(test_points)

# Our final results from our Test Points 
print("\nPredicted Steps for Test Points:")
for point, pred in zip(test_points.values, predictions):
    formatted_point = " ".join([f"{x:.4f}" for x in point])
    print(f"Input [{formatted_point}] = Predicted Step: {pred}")


