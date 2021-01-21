#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


# In[ ]:


##Opening the file containing data of Indian Liver Patients
liverdata = pd.read_csv("/Users/nidhihookeri/Downloads/Indian Liver Patient Dataset (ILPD).csv", sep=",")


# In[ ]:


About the dataset: 

This data set contains 416 liver patient records and 167 non liver patient records collected from North East 
of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient 
(liver disease) or not (no disease).This data set contains 441 male patient records and 142 female patient records.

The columns of the dataset are: 
    Age, Gender, Total bilirubin, Direct Bilirubin, Alkaline phosphatase, Alamine transferase,
    Aspartate aminotransferase, Total proteins, Albumin, Albumin and Globulin ratio, Dataset: a field used to split the dataset into 2 sets

is_patient constitutes as the dependent variable

The dataset split(is_patient): 
    1 indicates the presence of liver disease and 2 represents the absence of liver disease

The goal of this assignment:
    to evaluate the performance of regression (Linear regression) and classification (Decision tree) model with 
    respect to the dependent variable:'is_patient' based on important features selected from the dataset

Steps of CRISP-DM:
1) Business understanding: A simple blood report can contain numerous variables which might not be required for a 
    specific test. This assignment aims at gathering the important varaibles/features and modelling to further
    establish test runs on the selected data to understand the accuracy and fit of the model
2) Data understanding: Includes exploring data, undertanding the statistics of the dataset, variable-variable interaction
3) Data preparation: Includes treating the missing values, converting categorical into numerical values, feature selection 
using Pearsons correlation model to derive the important features for modelling 
4) Modelling: Linear Regression and Decision Tree 
5) Performance evaluation: MSE, MAE, RMSE and accuracy
6) Deployment: Focuses more on documentation of the process involved in selecting and implementating the algorithms
    
Reference: 
    https://www.kaggle.com/uciml/indian-liver-patient-records
    https://www.proglobalbusinesssolutions.com/six-steps-in-crisp-dm-the-standard-data-mining-process/
    


# In[ ]:


#DATA UNDERSTANDING
##Includes exploring data, verifying quality of data, checking the presence of NA values

##Converting the type of the data into a Panda DataFrame for easy accessing of columns
content = pd.DataFrame(liverdata)
##Accessing columns of the dataFrame
print(f"The columns of the dataset are as follows: {content.columns}")

##Checking the dimensions of the data
print(f"The dimensions of the dataset: {content.shape}")

##Checking the description of the data
#Gives information on the stats of dataset
#From this, we can derive information on missing data by relying on the row "count"
print(f"Description of the dataset: \n {content.describe()} \n")
#Result interpretation: In the column of alkaline phosphatase, there are 4 rows missing

##This helps in identifying numerical and categorical variables
print(content.info())

##Visualizing the dataset
###Interaction between the variables can be observed in order to understand if there are dependent/independent variables
scatter_matrix(content, figsize=(22,22))
plt.show


# In[ ]:


#DATA PREPARATION

#Omitting the rows that contain missing values; deleting 4 rows out of 583 wouldn't make a difference hence deleting the entire row instead of imputing missing data
liver_data = content.dropna()

#Converting categorical variable - gender into a numeric variable
#0 represents Female and 1 represents Male
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
x = liver_data['gender']
y = label_encoder.fit_transform(x)

#Feature Selection
#Best method for numeric input, numeric output is Pearson correlation
#Checking co-relation between the variables of the dataset
cor = liver_data.corr(method='pearson')

#Result Interpretation:
#-The 4 NA values are omitted leaving behind a total of 579 samples. 

#Result of correlation matrix and plot:
#According to the correlation matrix above, there is a co-relation between the direct_bilirubin and total_bilirubin, between sgot(Alanine Aminotransferase) and sgpt(Aspartate Aminotransferase) and between Alkaline Phosphatase and Aspartate aminotransferase. 
#The correlation between the direct and total is obvious, because as total bilirubin is the sum of direct and indirect bilirubin, so they have linear proportionality. 
#The relationship between sgpt and sgot also is strong since, both of them play a vital role in causing liver disease. Elevated levels of transferases cause liver disease. The correlation co-efficient is 0.78.
#Elevated levels of alkaline phosphatase also causes liver damage. The corelation coeffecient between alkaline phosphatase and Alanine Aminotransferase(sgot) is 0.69
#After studying the correlation between variables, we can say that, aspartate aminotransferase, alanine aminotransferase, alkaline phosphatase and direct bilirubin are leading factors in causing liver disease. 


# In[ ]:


#Scaling the data
from sklearn.preprocessing import MinMaxScaler
min_obj = MinMaxScaler()

num_data = liver_data[['direct_bilirubin', 'tot_proteins', 'albumin', 'sgpt', 'sgot', 'alkphos']]

scaled_data = min_obj.fit_transform(num_data).round(3)

scaled = pd.DataFrame(scaled_data, columns = num_data.columns)
print('The scaled data is:\n {}'.format(scaled))


# In[ ]:


#Splitting the dataset into train and test datasets
from sklearn.model_selection import train_test_split

#storing the label which is is_patient
x = liver_data.is_patient
#storing the labels 
y = scaled
split_dataset= train_test_split(y, x, test_size = 0.2)  #y is scaled data, x is labels
#print(split_dataset) #Four arrays are present. 
train = split_dataset[0]
#print(train)
test = split_dataset[1]
train_labels = split_dataset[2]
#print(train_labels)
test_labels = split_dataset[3]


# In[ ]:


#MODELLING AND PEROFRMANCE EVALUATION

#Linear model 
from sklearn import linear_model
linear_model_obj = linear_model.LinearRegression()
linear_data = linear_model_obj.fit(train, train_labels)

#Deriving the coef values with the feature names
coefs = pd.DataFrame(linear_data.coef_.round(2), train.columns, columns=['Co-effecients'])
print('{}\n'.format(coefs))

print('The intercept of the linear equation is {}\n'.format(linear_data.intercept_))

#Result
#From the .coef_ we derive the linear regression equation 
#y = mx + c
print(f"The linear equation would be: \n is_patient = -0.41*direct_bilirubin - 0.49*tot_proteins - 0.56*albumin - 0.46*sgpt + 0.72*sgot - 0.22*alkphos + 1.35")


# In[ ]:


#Performance Evaluation 

#Performing prediction on test data
prediction = linear_model_obj.predict(test).round(2)

comp = pd.DataFrame({'Actual' : test_labels, 'Predicted' : prediction})
#print('{}\n'.format(comp))

#Comparing the actual to the predicted values using plots
comp1 = comp.head(30)
print('{}\n'.format(comp1))
comp1.plot(kind='bar', figsize=(12,10))
mlt.pyplot.grid(which='major', linestyle='-', linewidth='0.5', color='black')
mlt.pyplot.xlabel("The row index of test vairable")
mlt.pyplot.ylabel("Labels")
mlt.pyplot.title("The comparison between actual and predicted values using linear regression")
mlt.pyplot.show()


# In[ ]:


#Evaluating the performance of the model
from sklearn import metrics
print('Mean Absolute Error (MAE) is', metrics.mean_absolute_error(test_labels, prediction).round(3))
print('Mean Squared Error (MSE) is', metrics.mean_squared_error(test_labels, prediction).round(3)) 
print('Root Mean Squared Error (RMSE) is', np.sqrt(metrics.mean_squared_error(test_labels, prediction)).round(3))

#Result Interpretation:
#Low MSE means that the predicted values are actually matching with the predicted values
#RMSE talks about how accurately the model fits the response. A low RMSE is an idicator of "good fit", which means the response is very close to the real values


# In[ ]:


#Decision Tree
from sklearn import tree
tree_obj = tree.DecisionTreeClassifier()
tree1 = tree_obj.fit(train, train_labels)

#Visualization of the tree
fig, ax = mlt.pyplot.subplots(figsize=(40,40))
tree.plot_tree(tree_obj, feature_names=liver_data.columns, class_names = True, filled=True, max_depth=4, fontsize=15)
mlt.pyplot.show()
mlt.pyplot.savefig('tree.png')


# In[ ]:


#Prediction
dec_prediction = tree1.predict(test)
comp2 = pd.DataFrame({'Actual': test_labels, 'Predicted':dec_prediction})
comp3 = comp2.head(25)
print('The comparison between test_labels and the prediction labels:\n {}'.format(comp3))

comp3.plot(kind='bar', figsize=(12,10))
mlt.pyplot.grid(which='major', linestyle='-', linewidth='0.5', color='black')
mlt.pyplot.xlabel("The row index of test variable")
mlt.pyplot.ylabel("Labels")
mlt.pyplot.title("The comparison between test_labels and prediction labels using Decision Tree")
mlt.pyplot.show()


# In[ ]:


#Performance evaluation 
acc = metrics.accuracy_score(test_labels, dec_prediction).round(3)
print('The accuracy of the model is {}'.format(acc))

#Result Interpretation:

#Accuracy is a way of evaluating classification models
#If the accuracy of the model is above the criteria of 0.5, then it can be considered in order to evaluate the model
#The accuracy of the model on this dataset is ~0.7 which supports the model to be fitting the dataset


# In[ ]:


The important features or variables that contribute to identifying the presence/absence of liver diease in patients 
are:
    1) aspartate aminotransferase
    2) alanine aminotransferase
    3) alkaline phosphatase
    4) direct bilirubin
These are further taken for modelling and performance evaluation

Comparing both the models:
    1) Linear Regression (regression oriented model)
    2) Decision tree (classification oriented model)

Regression models are evaluated effeciently in terms of MSE, MAE and RMSE. According to the results, as mentioned 
above the model seems to be a good fit for the chosen dataset. The regression line was fit in order to individually
get an idea around the presence or absence of liver disease. The MSE is quite low which supports the fact that the model fits good for the dataset.

Classification models are evaluated in terms of accuracy. The model showed about 0.69 accuracy which means a good fit
to the dataset. 69% of the test values are matching to the real values of the dataset.

Reference: 
    https://www.kaggle.com/uciml/indian-liver-patient-records

