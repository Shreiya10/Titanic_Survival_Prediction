#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 13:01:38 2023

@author: shreiyavenkatesan
"""

#importing required libraries for data analysis
import pandas as pd  
import numpy as np 
#importing required libraries for data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#%%
#Loading the train data set 
traindata = pd.read_csv('/Users/shreiyavenkatesan/Downloads/assignment2_train.csv')
#%%
#Train data Summary 
trainsummary = traindata.describe() 
#%%
#Data exploration in train data set 
plt.figure(figsize=(4, 5))
fig1= sns.countplot(x='Survived',hue='Survived', data=traindata,palette='PuOr')
fig1.set(xlabel ="Whether Survived or not", ylabel = "No. of persons", title ='Distribution of persons who survived')
plt.legend(labels = ['Not Survived', 'Survived'])
plt.show()

#Distribution of Male and Female Passengers
cols = ['pink' if x == "female" else 'lightblue' for x in traindata.Sex]
fig2=sns.countplot(x='Sex',data=traindata, palette= cols)
fig2.set(xlabel ="Gender", ylabel = "No. of persons", title ='Distribution of Male and Female Passengers')
plt.show()

##Comparison of Survival between Male V. Female
fig3 = sns.countplot(x='Sex', hue="Survived",data=traindata, palette= 'Pastel2')
fig3.set(xlabel ="Gender", ylabel = "Survival", title ='Survival- Male V. Female')
plt.show()

##Number of passengers survived-Passenger Classwise
fig4 = sns.countplot(x='Pclass', hue="Survived",data=traindata, palette= 'PuOr')
fig4.set(xlabel ="Passenger Class", ylabel = "Survival", title ='Number of passengers survived-Passenger Classwise')
plt.legend(labels = ['Not Survived', 'Survived'])
plt.show()

##Age Density of the Passengers
plt.figure(figsize = (15, 10))
sns.distplot(traindata['Age'].dropna(), color = (0, 0.5, 1), bins = 40, kde = True)
plt.title('Age Density of the Passengers', fontsize = 20)
plt.xlabel('Age', fontsize = 15)
plt.show()
#%%
traindata['Sex'] = [1 if x =='female' else 0 for x in traindata.Sex] 
#%%
traindata.replace({'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
#%%
plt.figure(figsize=(15,6))
sns.heatmap(traindata.corr(), vmax=1, square=True, annot=True)
#%%
#A BAR GRAPH TO SHOW THE FEATURES THAT ARE HIGHLY CORRELATED WITH SURVIVAL
plt.figure(figsize=(15,15))
traindata.corr()['Survived'].apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:11][::-1].plot(kind='barh',color=('darkgreen','maroon')) 
plt.title("CORRELATED FEATURES", size=20, pad=26)
plt.xlabel("Correlation Coefficients")
plt.ylabel("Features")
#%%
#Loading the test dataset 
testdata = pd.read_csv('/Users/shreiyavenkatesan/Downloads/assignment2_test-2.csv')
#%%
#train data summary 
testummary = testdata.describe() 
#%%
#Data exploration in test data set 

plt.figure(figsize=(4, 5))
fig5= sns.countplot(x='Survived',hue='Survived', data=testdata,palette='PuOr')
fig5.set(xlabel ="Whether Survived or not", ylabel = "No. of persons", title ='Distribution of persons who survived')
plt.legend(labels = ['Not Survived', 'Survived'])
plt.show()

#Distribution of Male and Female Passengers
cols = ['pink' if x == "female" else 'lightblue' for x in traindata.Sex]
fig6=sns.countplot(x='Sex',data=testdata, palette= cols)
fig6.set(xlabel ="Gender", ylabel = "No. of persons", title ='Distribution of Male and Female Passengers')
plt.show()

##Comparison of Survival between Male V. Female
fig7 = sns.countplot(x='Sex', hue="Survived",data=testdata, palette= 'Pastel2')
fig7.set(xlabel ="Gender", ylabel = "Survival", title ='Survival- Male V. Female')
plt.show()

##Number of passengers survived-Passenger Classwise
fig8 = sns.countplot(x='Pclass', hue="Survived",data=testdata, palette= 'PuOr')
fig8.set(xlabel ="Passenger Class", ylabel = "Survival", title ='Number of passengers survived-Passenger Classwise')
plt.legend(labels = ['Not Survived', 'Survived'])
plt.show()

##Age Density of the Passengers
plt.figure(figsize = (15, 10))
sns.distplot(testdata['Age'].dropna(), color = (0, 0.5, 1), bins = 40, kde = True)
plt.title('Age Density of the Passengers', fontsize = 20)
plt.xlabel('Age', fontsize = 15)
plt.show()
#%%
#dummies for Gender feature
testdata['Sex'] = [1 if x =='female' else 0 for x in testdata.Sex] 
#%%
#dummies for embarked feature
testdata.replace({'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
#%%
#Plotting a correlation matrix to check which features are related 
plt.figure(figsize=(15,6))
sns.heatmap(testdata.corr(), vmax=1, square=True, annot=True)
#%%
#A BAR GRAPH TO SHOW THE FEATURES THAT ARE HIGHLY CORRELATED WITH SURVIVAL
plt.figure(figsize=(15,15))
testdata.corr()['Survived'].apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:11][::-1].plot(kind='barh',color=('darkgreen','maroon')) 
plt.title("CORRELATED FEATURES", size=20, pad=26)
plt.xlabel("Correlation Coefficients")
plt.ylabel("Features")
#From the analysis of data it is evident that sex:female is perfectly correlated with survival in the test data set which makes it bias for prediction. Therefore I am combining both the data sets and then going to split it into train and test data
#%%
#merging the train and test data
data = pd.concat([traindata, testdata], ignore_index=True)
#%%
#Since travelling with siblings/spouse  or being parent child has some correlation with survival, I am combining the two features and seeing if the passenger is travelling ALONE or NOT. 
## Creation of new variable Is Alone to club : Parent Child and Siblings 
data['Single'] = data['SibSp'] + data['Parch']
#Distribution of number of persons travelling alone 
plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Single', data = data, palette='Spectral')
plt.title("WHETHER TRAVELLING ALONE OR NOT?", size=20, pad=26)
plt.xlabel('Single', fontsize = 15)
plt.ylabel('No. of Passengers', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.30, p.get_height() + 5))
plt.show()
#converting Travelling alone into a binary variable: Single V Not Single 
def convert_Single(data):
    
    bins = [None] * len(data)

    for i in range(len(data)):
        if(data.Single[i] in [0]):
            bins[i] = 'Single'
        if(data.Single[i] in [1, 2, 3, 4, 5, 6, 7, 10]):
            bins[i] = 'Not Single'

    data['Single'] = bins
    
convert_Single(data)

#Comparing Survival and whether the person is travelling alone or not 
plt.figure(figsize = (15, 10))
ax = sns.countplot(x = 'Single', hue = 'Survived', data = data,palette='ch:s=-.2,r=.6')
plt.title('Survival Count for the Single Feature', fontsize = 20)
plt.xlabel('Survived', fontsize = 15)
plt.ylabel('No. of persons', fontsize = 15)
for p in ax.patches:
        ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + 0.17, p.get_height() + 3))
plt.show()
#%%
#0 for not travelling single and 1 for travelling single 
data['Single'] = [1 if x =='Single' else 0 for x in data.Single] 

#%%
#dropping PARCH and SIBSP from the data set as it is subsumed into travelling single or not
data=data.drop(columns='Parch',axis=1)
data=data.drop(columns='SibSp',axis=1)
#dropping PassengerId,ticket type,cabin number and name as they are not relevant to prediction and do not have any role to predict if the passenger would survive or not
data= data.drop(columns='Cabin', axis=1)
data= data.drop(columns='Name', axis=1)
data= data.drop(columns='PassengerId', axis=1)
data= data.drop(columns='Ticket', axis=1)
#dropping Fare from data as Class of ticket can indicate the fare that the passenger must have paid
data= data.drop(columns='Fare', axis=1)
#%%
#checking for null values 
data.isnull().sum()
data.info()
#%% 
#dropping NA values in age , fare and embarked 
#data.dropna(inplace=True)
#Filling the missing values in age with mean values as it does not significantly affect the dataset
data['Age'].fillna(data['Age'].mean(), inplace=True)
data.isnull().sum()
#%%
#filling missing embarked value with mode
print(data['Embarked'].mode())
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
#%%
plt.figure(figsize=(15,15))
data.corr()['Survived'].apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:11][::-1].plot(kind='barh',color=('purple','pink')) 
plt.title("CORRELATED FEATURES", size=20, pad=26)
plt.xlabel("Correlation Coefficients")
plt.ylabel("Features")
#%%
#Survival of Male and Female based on combined data set
fig10 = sns.countplot(x='Sex', hue="Survived",data=data, palette= 'Spectral')
fig10.set(xlabel ="Gender", ylabel = "Survival", title ='Survival- Male V. Female')
plt.show()
#%%
##importing the decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
#%%
#Scaling  Age 
numeric_features = ['Age']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])
#%%
features = ['Pclass','Age','Sex','Embarked','Single']
label = ['Survived']
x = data.loc[:,features]
y = data.loc[:,['Survived']]
#%%
#Splitting into train and test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y , 
                          random_state=42,
                          train_size=0.8,shuffle=(True))
#%%
#Fitting the model in the train data
tree = DecisionTreeClassifier(random_state=0)
tree.fit(xtrain, ytrain)
#%%
#checking the accuracy on test data set
print("Accuracy on test set: {:.3f}".format(tree.score(xtest, ytest)))
#%%
#Plotting the decision tree
from sklearn.tree import plot_tree
plot_tree(tree, 
          feature_names = 'features', 
          class_names = 'label', 
          filled = True, 
          rounded = True)
plt.savefig('tree_visualization.png') 
#%%
#cost complexity pruning 
path = tree.cost_complexity_pruning_path(xtrain, ytrain)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#%%
#estimating a tree for each alpha 
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(xtrain, ytrain)
    clfs.append(clf)
#%%
#dropping the last model as it has only one node
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
#%%
##plotting accuracy (in test and training) over alpha; first compute accuracy for each alpha
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(xtrain)
    y_test_pred = c.predict(xtest)
    train_acc.append(accuracy_score(y_train_pred,ytrain))
    test_acc.append(accuracy_score(y_test_pred,ytest))
#%%
plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()
#%%
##Estimating the tree with optimal alpha 
clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.0025)
clf_.fit(xtrain,ytrain)
print("Accuracy on test set: {:.3f}".format(clf_.score(xtest, ytest)))
#%%
##plotting the pruned version of the tree
pic = plt.figure(figsize=(30,15))
plot_tree(clf_, 
          feature_names = ['Pclass','Age','Sex','Single','Embarked'], 
          class_names = ['Died', 'Survived'], 
          filled = True, 
          rounded = True,)

plt.savefig('tree_visualization.png') 
#%%
pred_2 = clf_.predict(xtest)
y_prob=clf_.predict_proba(xtest)
matrix1 = confusion_matrix(pred_2, ytest)
names = np.unique(pred_2)
matrix1 = pd.DataFrame(matrix1, index=np.unique(ytest), columns=np.unique(pred_2))
matrix1.index.name = 'Actual'
matrix1.columns.name = 'Predicted'
DT=sns.heatmap(matrix1, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names,cmap="YlGnBu")
plt.title("Decision Tree Confusion Matrix")

#%%
#PROBIT MODEL 
import statsmodels.api as smf
#%%
features = ['Pclass','Age','Sex','Single','Embarked']
label = ['Survived']
x = data.loc[:,features]
y = data.loc[:,['Survived']]
#%%
#Fitting the model
mod=smf.Probit(ytrain,xtrain).fit()
print(mod.summary())
#%%
#prediction 
predy = mod.predict(xtest.astype(float))
print(predy) 
#%%
#accuracy
y_pred_binary = [1 if y>= 0.5 else 0 for y in predy]
accuracy_score(ytest,y_pred_binary)
#%%
pred_3 = mod.predict(xtest.astype(float))
y_pred_binary = [1 if y>= 0.5 else 0 for y in pred_3]
matrix2 = confusion_matrix(y_pred_binary, ytest)
names = np.unique(y_pred_binary)
matrix2 = pd.DataFrame(matrix2, index=np.unique(ytest), columns=np.unique(y_pred_binary))
matrix2.index.name = 'Actual'
matrix2.columns.name = 'Predicted'
sns.heatmap(matrix2, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names,cmap="Accent")
plt.title("Probit Confusion Matrix")