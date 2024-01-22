#!/usr/bin/env python
# coding: utf-8

# **Required libraries**

# In[11]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import binarize, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# **Functions**

# In[12]:


def encoder(dataset, catFeatures, qtyFeatures):
  dataset = dataset[catFeatures + qtyFeatures]
  dataset_encoded = pd.get_dummies(dataset, 
                                   columns = catFeatures, 
                                   drop_first = True)
  
  return(dataset_encoded)

def plot_auc_curve(model, X, y):
  try:
      y_pred_prob = model.predict_proba(X)[:,1]
  except:
    d = model.decision_function(X)
    y_pred_prob = np.exp(d) / (1 + np.exp(d))
  
  auc = roc_auc_score(y, y_pred_prob)
  fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
  
  plt.plot(fpr, tpr)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.title('ROC Curve\n AUC={auc}'.format(auc = auc))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid(True)

def model_training(model, X, y):
  model.fit(X,y)
  
  return(model)

def print_accurcay_metrics(model, X, y, threshold):
  try:
      y_pred_prob = model.predict_proba(X)[:,1]
  except:
    d = model.decision_function(X)
    y_pred_prob = np.exp(d) / (1 + np.exp(d))
  y_pred_class = binarize([y_pred_prob], threshold)[0]
  
  print("Accurcay:", accuracy_score(y, y_pred_class))
  print("AUC:", roc_auc_score(y, y_pred_prob))
  print("Log Loss:", log_loss(y, y_pred_prob))
  print("Confusion Matrix:\n", confusion_matrix(y, y_pred_class))
  print("Recall:", recall_score(y, y_pred_class))
  print("Precision:", precision_score(y, y_pred_class))


def Find_Optimal_Cutoff(model, X, y):
  try:
    y_pred_prob = model.predict_proba(X)[:,1]
  except:
    d = model.decision_function(X)
    y_pred_prob = np.exp(d) / (1 + np.exp(d))
    
  fpr, tpr, threshold = roc_curve(y, y_pred_prob)
  i = np.arange(len(tpr)) 
  roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
  roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
  
  print("Optimal Cutoff:", roc_t['threshold'].values)
  return(roc_t['threshold'].values)
  
def feature_importance(model, X):
  importances = model.feature_importances_
  std = np.std([tree.feature_importances_ for tree in model.estimators_],
               axis=0)
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  for f in range(X.shape[1]):
      print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

  # Plot the feature importances of the forest
  plt.figure()
  plt.title("Feature importances")
  plt.bar(range(X.shape[1]), importances[indices],
          color="r", yerr=std[indices], align="center")
  plt.xticks(range(X.shape[1]), indices)
  plt.xlim([-1, X.shape[1]])
  plt.show()

# Plot calibration plots
def plot_calibration(y_true, y_prob, n_bins, model_name):
  plt.figure(figsize=(10, 10))
  ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
  ax2 = plt.subplot2grid((3, 1), (2, 0))

  ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
  fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_true, y_prob, n_bins=n_bins)

  ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (model_name, ))

  ax2.hist(y_pred_prob, range=(0, 1), bins=10, label=model_name,
             histtype="step", lw=2)

  ax1.set_ylabel("Fraction of positives")
  ax1.set_ylim([-0.05, 1.05])
  ax1.legend(loc="lower right")
  ax1.set_title('Calibration plots')

  ax2.set_xlabel("Mean predicted value")
  ax2.set_ylabel("Count")
  ax2.legend(loc="upper right", ncol=2)

  plt.tight_layout()
  plt.show()


# **Confusion Matrix**
# *   first argument is actual values, second argument is predicted values
# 
# **Precision:**
# *  When a positive value is predicted, how often is the prediction correct?
# *  How "precise" is the classifier when predicting positive instances?
# 
# **Recall**
# *   Calculate TP rate or recall
# 
# 

# **Pandas**
# *   use the pandas library to read data into Python
# 

# In[18]:


file_url ='https://drive.google.com/uc?export=download&id=13YvNpCckww8rNdLGxAh0tekdsehXI0HU'
df = pd.read_csv(file_url)


# Use the head method to display the first 5 rows

# In[19]:


df.head()


# check the shape of the DataFrame (rows, columns)

# In[20]:


df.shape


# In[21]:


df.dtypes


# Use the describe method to generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values

# In[22]:


df.describe()


# In[23]:


df.loc[((df['Credit_Score'] > 850) & 
        (df['Credit_Score'] < 300)), 'Credit_Score'] = np.nan


# Count the number of NA values for each variable

# In[24]:


df.isnull().sum()


# Number of examples per class

# In[25]:


df.groupby('Loan_Status')['Loan_ID'].count()


# In[26]:


loanIDCount = df.groupby('Loan_ID')['Loan_Status'].count()
loanIDCount = loanIDCount.sort_values(ascending = False)
loanIDCount[0:5]


# #### Checking for the duplicate LOAN_ID

# In[27]:


df[df['Loan_ID'] == 'ffffe32e-ed17-459f-9cfd-7b9ee7972933']


# #### Remove duplicates

# In[28]:


df.drop_duplicates(inplace = True)


# In[29]:


df.shape


# In[30]:


loanIDCount = df.groupby('Loan_ID')['Loan_Status'].count()
loanIDCount = loanIDCount.sort_values(ascending = False)
loanIDCount[0:5]


# In[31]:


df[df['Loan_ID'] == '7f85733b-ef51-48e5-a0a2-1a7b68492d4e']


# In[32]:


df['naCount'] = df.isnull().sum(axis=1)


# In[33]:


df.head()


# In[34]:


df.sort_values(by=['Loan_ID', 'naCount'], inplace = True)


# In[35]:


df.drop_duplicates(subset=['Loan_ID'], keep = 'first', inplace = True)


# In[36]:


loanIDCount = df.groupby('Loan_ID')['Loan_Status'].count()
loanIDCount = loanIDCount.sort_values(ascending = False)
loanIDCount[0:5]


# In[37]:


print(len(df['Loan_ID'].unique()))
print(df.shape)


# In[38]:


df.isnull().sum()


# In[39]:


df.describe()


# In[40]:


df.dtypes


# In[41]:


yearsInCurrentJobCount = df.groupby('Years_in_current_job')['Loan_ID'].count()
yearsInCurrentJobCount = yearsInCurrentJobCount.sort_values(ascending = False)
yearsInCurrentJobCount


# #### Rounding off the columns

# In[42]:


df.loc[(df['Years_in_current_job'] == '< 1 year'), 'Years_in_current_job'] = 0.5
df.loc[(df['Years_in_current_job'] == '1 year'), 'Years_in_current_job'] = 1
df.loc[(df['Years_in_current_job'] == '2 years'), 'Years_in_current_job'] = 2
df.loc[(df['Years_in_current_job'] == '3 years'), 'Years_in_current_job'] = 3
df.loc[(df['Years_in_current_job'] == '4 years'), 'Years_in_current_job'] = 4
df.loc[(df['Years_in_current_job'] == '5 years'), 'Years_in_current_job'] = 5
df.loc[(df['Years_in_current_job'] == '6 years'), 'Years_in_current_job'] = 6
df.loc[(df['Years_in_current_job'] == '7 years'), 'Years_in_current_job'] = 7
df.loc[(df['Years_in_current_job'] == '8 years'), 'Years_in_current_job'] = 8
df.loc[(df['Years_in_current_job'] == '9 years'), 'Years_in_current_job'] = 9
df.loc[(df['Years_in_current_job'] == '10+ years'), 'Years_in_current_job'] = 10


# In[43]:


df['Years_in_current_job'] = df['Years_in_current_job'].astype('float')


# In[44]:


monthlyDebtCount = df.groupby('Monthly_Debt')['Loan_ID'].count()
monthlyDebtCount = monthlyDebtCount.sort_values(ascending = False)
monthlyDebtCount


# Convert number strings with commas to float

# In[45]:


df["Monthly_Debt"] = df["Monthly_Debt"].str.replace(",","").astype(float)
df['Monthly_Debt'] = df['Monthly_Debt'].astype('float')


# In[46]:


df.dtypes


# In[47]:


df.isnull().sum()


# Fill in missing values with mean

# In[48]:


for f in ['Credit_Score', 
          'Years_in_current_job', 
          'Annual_Income', 
          'Months_since_last_delinquent',
          'Bankruptcies', 'Tax_Liens']:
  df.loc[df[f].isnull(), f] = df[f].mean()


# In[49]:


df.isnull().sum()


# In[50]:


df.dtypes


# In[51]:


qtyFeatures = ['Current_Loan_Amount', 
               'Credit_Score', 'Years_in_current_job', 
               'Annual_Income', 'Monthly_Debt', 
               'Years_of_Credit_History',
               'Months_since_last_delinquent', 'Number_of_Open_Accounts',
               'Current_Credit_Balance',
               'Maximum_Open_Credit', 
               'Number_of_Credit_Problems', 'Bankruptcies', 'Tax_Liens'
              ]

catFeatures = ['Term', 'Home_Ownership', 'Purpose']

label = 'Loan_Status'


# In[52]:


df.groupby(['Home_Ownership'])['Loan_ID'].count()


# In[53]:


df.loc[(df['Home_Ownership'] == 'HaveMortgage'), 'Home_Ownership'] = 'Home Mortgage' 


# In[54]:


df.groupby(['Term'])['Loan_ID'].count()


# In[55]:


df.groupby(['Purpose'])['Loan_ID'].count()


# In[56]:


df.loc[(df['Purpose'] == 'other'), 'Purpose'] = 'Other' 


# In[57]:


df['Loan_Status'].unique()


# **Preparing X and y using pandas:**

# In[58]:


X_encoded = encoder(df, catFeatures, qtyFeatures)

le = LabelEncoder()
y_encoded = le.fit_transform(df[label])


# In[59]:


X_encoded.head()


# In[60]:


y_encoded


# Splitting X and y into training and testing sets

# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X_encoded, 
                                                    y_encoded, 
                                                    test_size = 0.2,
                                                    random_state = 1)


# **Fit logistic regression**

# In[62]:


logreg = LogisticRegression()
model_training(logreg, X_train, y_train)

plot_auc_curve(logreg, X_test, y_test)
Find_Optimal_Cutoff(logreg, X_test, y_test)
print_accurcay_metrics(logreg, X_test, y_test, 0.65)


# In[ ]:


plot_auc_curve(logreg, X_train, y_train)
Find_Optimal_Cutoff(logreg, X_train, y_train)
print_accurcay_metrics(logreg, X_train, y_train, 0.65)


# In[ ]:


y_pred_prob = logreg.predict_proba(X_test)[:, 1]

plot_calibration(y_test, y_pred_prob, 20, "Logistic Regression")


# In[ ]:


y_pred_prob = logreg.predict_proba(X_train)[:, 1]

plot_calibration(y_train, y_pred_prob, 20, "Logistic Regression")


# In[ ]:


print_accurcay_metrics(logreg, X_train, y_train, 0.65)


# **Fit Logistic regression with L2 regularization (Ridge)**

# In[ ]:


ridge = RidgeClassifier(alpha=1.0)
model_training(ridge, X_train, y_train)

plot_auc_curve(ridge, X_test, y_test)
Find_Optimal_Cutoff(ridge, X_test, y_test)
print_accurcay_metrics(ridge, X_test, y_test, 0.61)


# In[ ]:


y_pred_prob = logreg.predict_proba(X_test)[:, 1]

plot_calibration(y_test, y_pred_prob, 20, "Ridge")


# In[ ]:


alpha_range = np.arange(0.001, 5, 0.1)
scores = []
for lam in alpha_range:
  ridge = RidgeClassifier(alpha=lam)
  scores.append(cross_val_score(ridge, X_train, y_train, cv=5, scoring='roc_auc').mean())

plt.plot(alpha_range, scores)
plt.xlim([0.0, 5])
#plt.ylim([0.0, 1.0])
plt.title('L2 Regularization')
plt.xlabel('Lambda')
plt.ylabel('AUC')
plt.grid(True)


# In[63]:


scores


# **Fit Logistic regression with L1 regularization (Lasso)**

# In[64]:


lasso = LogisticRegression(penalty='l1', C=1)
model_training(lasso, X_train, y_train)

plot_auc_curve(lasso, X_test, y_test)
Find_Optimal_Cutoff(lasso, X_test, y_test)
print_accurcay_metrics(lasso, X_test, y_test, 0.73)


# In[ ]:


lambda_range = np.arange(0.001, 5, 0.1)
scores = []
for lam in lambda_range:
  lasso = LogisticRegression(penalty='l1', C=1/lam)
  scores.append(cross_val_score(lasso, X_train, y_train, cv=5, scoring='roc_auc').mean())



# In[65]:


plt.plot(lambda_range, scores)
plt.xlim([0.0, 5])
#plt.ylim([0.0, 1.0])
plt.title('L1 Regularization')
plt.xlabel('Lambda')
plt.ylabel('AUC')
plt.grid(True)


# **2-degree Polynomial Features**

# In[66]:


poly = PolynomialFeatures(2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)


# In[67]:


print(X_test.shape)
print(X_test_poly.shape)


# In[68]:


logreg_poly = LogisticRegression()
model_training(logreg_poly, X_train_poly, y_train)

plot_auc_curve(logreg_poly, X_test_poly, y_test)
Find_Optimal_Cutoff(logreg_poly, X_test_poly, y_test)
print_accurcay_metrics(logreg_poly, X_test_poly, y_test, 0.75)


# In[ ]:


plot_auc_curve(logreg_poly, X_train_poly, y_train)
Find_Optimal_Cutoff(logreg_poly, X_train_poly, y_train)
print_accurcay_metrics(logreg_poly, X_train_poly, y_train, 0.75)


# In[ ]:


print_accurcay_metrics(logreg_poly, X_train_poly, y_train, 0.75)


# In[ ]:


ridge_poly = RidgeClassifier(alpha=1.0)
model_training(ridge_poly, X_train_poly, y_train)

plot_auc_curve(ridge_poly, X_test_poly, y_test)
Find_Optimal_Cutoff(ridge_poly, X_test_poly, y_test)
print_accurcay_metrics(ridge_poly, X_test_poly, y_test, 0.6)


# In[ ]:


lasso_poly = LogisticRegression(penalty='l1', C=1)
model_training(lasso_poly, X_train_poly, y_train)

plot_auc_curve(lasso_poly, X_test_poly, y_test)
Find_Optimal_Cutoff(lasso_poly, X_test_poly, y_test)
print_accurcay_metrics(lasso_poly, X_test_poly, y_test, 0.73)


# **Ranfom Forest**

# In[69]:


import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 123

X = X_encoded
y = y_encoded

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(n_estimators=100,
                               warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 1
max_estimators = 10

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()


# In[70]:


rf = RandomForestClassifier(n_estimators=100, 
                            criterion='gini', 
                            max_features='sqrt',
                            n_jobs=-1)
model_training(rf, X_train, y_train)

plot_auc_curve(rf, X_test, y_test)
Find_Optimal_Cutoff(rf, X_test, y_test)
print_accurcay_metrics(rf, X_test, y_test, 0.73)


# In[ ]:


rf = RandomForestClassifier(n_estimators=100, 
                            criterion='gini', 
                            max_features=None,
                            n_jobs=-1)
model_training(rf, X_train, y_train)

plot_auc_curve(rf, X_test, y_test)
Find_Optimal_Cutoff(rf, X_test, y_test)
print_accurcay_metrics(rf, X_test, y_test, 0.73)


# In[ ]:


print_accurcay_metrics(rf, X_train, y_train, 0.73)


# **Feature Importances**

# In[ ]:


feature_importance(rf, X_train)


# In[ ]:


rf_poly = RandomForestClassifier(n_estimators=100, 
                            criterion='gini', 
                            max_features='sqrt',
                            n_jobs=-1)
model_training(rf_poly, X_train_poly, y_train)

plot_auc_curve(rf_poly, X_test_poly, y_test)
Find_Optimal_Cutoff(rf_poly, X_test_poly, y_test)
print_accurcay_metrics(rf_poly, X_test_poly, y_test, 0.73)


# **Bagging**

# In[ ]:


bagging = RandomForestClassifier(n_estimators=10, 
                            criterion='gini', 
                            max_features=None,
                            n_jobs=-1)
model_training(bagging, X_train, y_train)

plot_auc_curve(bagging, X_test, y_test)
Find_Optimal_Cutoff(bagging, X_test, y_test)
print_accurcay_metrics(bagging, X_test, y_test, 0.73)


# In[ ]:


print_accurcay_metrics(bagging, X_train, y_train, 0.73)


# **Boosting**

# In[ ]:


adaBoost = AdaBoostClassifier(n_estimators=100)
model_training(adaBoost, X_train, y_train)

plot_auc_curve(adaBoost, X_test, y_test)
Find_Optimal_Cutoff(adaBoost, X_test, y_test)
print_accurcay_metrics(adaBoost, X_test, y_test, 0.73)


# In[ ]:


n_range = np.arange(100, 500, 100)
scores = []
for n in n_range:
  adaBoost = AdaBoostClassifier(n_estimators=n)
  scores.append(cross_val_score(adaBoost, X_train, y_train, cv=5, scoring='roc_auc').mean())


plt.plot(n_range, scores)
plt.xlim([0.0, 500])
#plt.ylim([0.0, 1.0])
plt.title('AdaBoostClassifier')
plt.xlabel('n_estimators')
plt.ylabel('AUC')
plt.grid(True)


# In[ ]:


n_range = np.arange(10, 100, 10)
scores = []
for n in n_range:
  adaBoost = AdaBoostClassifier(n_estimators=n)
  scores.append(cross_val_score(adaBoost, X_train, y_train, cv=5, scoring='roc_auc').mean())


plt.plot(n_range, scores)
plt.xlim([0.0, 100])
#plt.ylim([0.0, 1.0])
plt.title('AdaBoostClassifier')
plt.xlabel('n_estimators')
plt.ylabel('AUC')
plt.grid(True)


# In[ ]:


plt.plot(n_range, scores)
plt.xlim([0.0, 200])
#plt.ylim([0.0, 1.0])
plt.title('AdaBoostClassifier')
plt.xlabel('n_estimators')
plt.ylabel('AUC')
plt.grid(True)


# In[81]:


import streamlit as st

st.write("""
# My first app
Hello *world!*
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




