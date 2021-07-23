# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
#from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import NearMiss
#from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('creditcard.csv')
df.head()
#Skew of classes
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
#Sns bar plot
colors = ["#0101DF", "#DF0101"]
sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: Non Frauduleux || 1: Frauduleux)', fontsize=14)
#Scaling
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)
#---------------------------------------------------#
# Spliting the Data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

len(df.loc[df["Class"]==0])
len(df.loc[df["Class"]==1])

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
#original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

#hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
#dt=DecisionTreeClassifier(criterion=hdc)
#dt.fit(original_Xtrain, original_ytrain)
#from sklearn.model_selection import cross_val_score
#training_score = cross_val_score(dt,original_Xtrain , original_ytrain, cv=5)
#print("Classifiers: DecisionTree with HDDT", "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Check the Distribution of the labels


# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))
#---------------------------------------------------#
#Undersample then Shuffle our Data
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()
#---------------------------------------------------#
#Plot des données équilibrées
print('Distribution des classes de l''ensemble d''apprentissage')
print(new_df['Class'].value_counts()/len(new_df))
sns.countplot('Class', data=new_df, palette=colors)
plt.title('Distibution égale', fontsize=14)
plt.show()
#---------------------------------------------------#
#Classification phase Original case

#Classification phase Undersampled case
X = new_df.drop('Class', axis=1)
y = new_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
from hellinger_distance_criterion import HellingerDistanceCriterion
#hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
#clf = RandomForestClassifier(criterion=hdc, max_depth=4, n_estimators=100)
#clf.fit(X_train, y_train)
#print('hellinger distance score: ', clf.score(X_test, y_test))
hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
dt=DecisionTreeClassifier(criterion=hdc)
dt.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
training_score = cross_val_score(dt, X_train, y_train, cv=5)
print("Classifiers: DecisionTree with Hellinger", "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
print('-'*10)
#---------------------------------------------------------------#
#Classification of different type of Decision Tree
#classifiers = {
#    "HDDT": DecisionTreeClassifier(criterion=hdc),
#    "Support Vector Classifier": SVC(),
#    "DecisionTreeClassifier C4.5": DecisionTreeClassifier(criterion="entropy"),
#    "DecisionTreeClassifier Gini": DecisionTreeClassifier()
#}
classifiers = {
    "HDDT": DecisionTreeClassifier(criterion=hdc),
    "DecisionTreeClassifier C4.5": DecisionTreeClassifier(criterion="entropy"),
    "DecisionTreeClassifier Gini": DecisionTreeClassifier()
}

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    #training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    tree_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__,"Has a training score of", round(tree_score.mean() * 100, 2).astype(str) + '%')
    #print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
#---------------------------------------------------------------#
#ROC Curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.
dt1=DecisionTreeClassifier(criterion=hdc)
dt1.fit(X_train, y_train)
#classifier.fit(X_train, y_train)
#training_score = cross_val_score(classifier, X_train, y_train, cv=5)
dt2=DecisionTreeClassifier()
dt2.fit(X_train, y_train)

dt3=DecisionTreeClassifier(criterion="entropy")
dt3.fit(X_train, y_train)
#--------------------------------------------------------------------------------------------#
#Résumé des scores en training voir si il y a effet de overfitting
HDDT = cross_val_score(dt1, X_train, y_train, cv=5)
Gini = cross_val_score(dt2, X_train, y_train, cv=5)
IG = cross_val_score(dt3, X_train, y_train, cv=5)
#-------------------------------------------------------------------------#
#Prediction of all models

#log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,method="decision_function")
#knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)
#svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,method="decision_function")
#tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

HDDT_pred = cross_val_predict(dt1, X_train, y_train, cv=5)
Gini_pred = cross_val_predict(dt2, X_train, y_train, cv=5)
IG_pred = cross_val_predict(dt3, X_train, y_train, cv=5)


#log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
#knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
#svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)

from sklearn.metrics import roc_auc_score
print('HDDT: ', roc_auc_score(y_train, HDDT_pred))
print('Gini: ', roc_auc_score(y_train, Gini_pred))
print('C4.5: ', roc_auc_score(y_train, IG_pred))



dt1_fpr, dt1_tpr, dt1_threshold = roc_curve(y_train, HDDT_pred)
dt2_fpr, dt2_tpr, dt2_threshold = roc_curve(y_train, Gini_pred)
dt3_fpr, dt3_tpr, dt3_threshold = roc_curve(y_train, IG_pred)


def graph_roc_curve_multiple(dt1_fpr, dt1_tpr, dt2_fpr, dt2_tpr, dt3_fpr, dt3_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(dt1_fpr, dt1_tpr, label='HDDT Score: {:.4f}'.format(roc_auc_score(y_train, HDDT_pred)))
    plt.plot(dt2_fpr, dt2_tpr, label='Gini Score: {:.4f}'.format(roc_auc_score(y_train, Gini_pred)))
    plt.plot(dt3_fpr, dt3_tpr, label='C4.5 Score: {:.4f}'.format(roc_auc_score(y_train, IG_pred)))
    #plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score pour 50% \n (C''est le score minimum qu''on peut avoir)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(dt1_fpr, dt1_tpr, dt2_fpr, dt2_tpr, dt3_fpr, dt3_tpr)
plt.show()
#------------------------------------------------------------------#
#Precision Recall etc..
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, HDDT_pred)

y_pred = dt3.predict(X_train)

# Overfitting Case
print('---' * 45)
print('Overfitting: \n')
print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print('---' * 45)

# How it should look like
#print('---' * 45)
#print('How it should be:\n')
#print("Accuracy Score: {:.2f}".format(np.mean(undersample_accuracy)))
#print("Precision Score: {:.2f}".format(np.mean(undersample_precision)))
#print("Recall Score: {:.2f}".format(np.mean(undersample_recall)))
#print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))
#print('---' * 45)

#------------------------------------------------------------#
#Confusion Matrix for all classifiers
print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

from sklearn.metrics import confusion_matrix

# Logistic Regression fitted using SMOTE technique
y_pred_dt1 = dt1.predict(X_test)

# Other models fitted with UnderSampling
y_pred_dt2 = dt2.predict(X_test)
y_pred_dt3 = dt3.predict(X_test)
#☻y_pred_tree = tree_clf.predict(X_test)


HDDT_cf = confusion_matrix(y_test, y_pred_dt1)
Gini_cf = confusion_matrix(y_test, y_pred_dt2)
IG_cf = confusion_matrix(y_test, y_pred_dt3)
#tree_cf = confusion_matrix(y_test, y_pred_tree)

fig, ax = plt.subplots(2,2,figsize=(20,12))


sns.heatmap(HDDT_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("HDDT classifier \n Matrice de confusion", fontsize=15)
ax[0, 0].set_xticklabels(['', ''], fontsize=20, rotation=80)
ax[0, 0].set_yticklabels(['', ''], fontsize=10, rotation=200)

sns.heatmap(Gini_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
ax[0][1].set_title("Gini classifier \n Matrice de confusion", fontsize=15)
ax[0][1].set_xticklabels(['', ''], fontsize=10, rotation=80)
ax[0][1].set_yticklabels(['', ''], fontsize=10, rotation=200)

sns.heatmap(IG_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
ax[1][0].set_title("C4.5 Classifier \n Matrice de confusion", fontsize=15)
ax[1][0].set_xticklabels(['', ''], fontsize=20, rotation=80)
ax[1][0].set_yticklabels(['', ''], fontsize=20, rotation=200)

#sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)
#ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
#ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
#ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

plt.show()





from sklearn.metrics import classification_report


print('HDDT Classifier:')
print(classification_report(y_test, y_pred_dt1))

print('Gini Classifier:')
print(classification_report(y_test, y_pred_dt2))

print('C4.5 Classifier:')
print(classification_report(y_test, y_pred_dt3))

#print('Support Vector Classifier:')
#print(classification_report(y_test, y_pred_tree))


