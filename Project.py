'''
extraColumns = ['REC_ID','EVENT_ID','PAG_NAME','INFODT','ORIG_ENTRY','LAST_UPDATE']
verbalData = verbalData.drop(extraColumns, axis=1)
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:52:02 2022

@author: Arpit Mittal
"""
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset
from tensorflow.keras.utils import to_categorical
import numpy as np
import seaborn
import warnings
warnings.filterwarnings("ignore")

print("----(Data Preprocessing...)----")
labelData =  pd.read_csv('Conclusion_of_Study_Participation.csv', sep=",")
labeledPatientsint = labelData["PATNO"].tolist()
labeledPatients = [str(x) for x in labeledPatientsint]

lexicalData =  pd.read_csv('Lexical_Fluency.csv', sep=",")
lexicalDataFiltered = lexicalData[lexicalData['PATNO'].isin(labeledPatients)].fillna(0).astype(int)

'''
------------
'''
motorMvtData =  pd.read_csv('MDS_UPDRS_Part_III.csv', sep=",")
motorMvtDataFiltered = motorMvtData[motorMvtData['PATNO'].isin(labeledPatientsint)].fillna(0).astype(int)
motorMvtDataFiltered=motorMvtDataFiltered.groupby(['PATNO']).mean().reset_index()
motorMvtDataPatients = motorMvtDataFiltered["PATNO"].tolist()

verbalData =  pd.read_csv('Hopkins_Verbal_Learning_Test.csv', sep=",")
verbalDataFiltered = verbalData[verbalData['PATNO'].isin(labeledPatients)].fillna(0).astype(int)
verbalDataFiltered=verbalDataFiltered.groupby(['PATNO']).mean().reset_index()
verbalDataPatients = verbalDataFiltered["PATNO"].tolist()

semanticData =  pd.read_csv('Semantic_Fluency.csv', sep=",")
semanticDataFiltered = semanticData[semanticData['PATNO'].isin(labeledPatients)].fillna(0).astype(int)
semanticDataFiltered=semanticDataFiltered.groupby(['PATNO']).mean().reset_index()
semanticDataPatients = semanticDataFiltered["PATNO"].tolist()

'''
------------
'''
labelDataFiltered = labelData[labelData['PATNO'].isin(verbalDataPatients)].fillna(0).astype(int)
labelDataFiltered = labelData[labelData['PATNO'].isin(motorMvtDataPatients)].fillna(0).astype(int)
labelDataFiltered = labelData[labelData['PATNO'].isin(semanticDataPatients)].fillna(0).astype(int)

'''
------------
'''

labelDataFiltered = labelDataFiltered[labelDataFiltered['PATNO'].isin(verbalDataPatients)].fillna(0).astype(int)
labelDataFiltered = labelDataFiltered[labelDataFiltered['PATNO'].isin(motorMvtDataPatients)].fillna(0).astype(int)
labelDataFiltered = labelDataFiltered[labelDataFiltered['PATNO'].isin(semanticDataPatients)].fillna(0).astype(int)
labeledPatientsint = labelDataFiltered["PATNO"].tolist()
labeledPatients = [str(x) for x in labeledPatientsint]

motorMvtDataFiltered = motorMvtDataFiltered[motorMvtDataFiltered['PATNO'].isin(labeledPatientsint)].fillna(0).astype(int)
motorMvtDataFiltered = motorMvtDataFiltered.groupby(['PATNO']).mean().reset_index()
motorMvtDataPatients = motorMvtDataFiltered["PATNO"].tolist()

verbalData =  pd.read_csv('Hopkins_Verbal_Learning_Test.csv', sep=",")
verbalDataFiltered = verbalData[verbalData['PATNO'].isin(labeledPatients)].fillna(0).astype(int)
verbalDataFiltered=verbalDataFiltered.groupby(['PATNO']).mean().reset_index()
verbalDataPatients = verbalDataFiltered["PATNO"].tolist()

semanticData =  pd.read_csv('Semantic_Fluency.csv', sep=",")
semanticDataFiltered = semanticData[semanticData['PATNO'].isin(labeledPatients)].fillna(0).astype(int)
semanticDataFiltered = semanticDataFiltered.groupby(['PATNO']).mean().reset_index()
semanticDataPatients = semanticDataFiltered["PATNO"].tolist()

'''
------------
'''

# In[2]
completeDataList = [verbalDataFiltered,semanticDataFiltered,motorMvtDataFiltered,labelDataFiltered]
finalDataSet = reduce(lambda left,right: pd.merge(left,right,on='PATNO'), completeDataList)
print("Complete Annotated Dataset : ")
print(finalDataSet)

completeFeaturesDataList = [verbalDataFiltered,semanticDataFiltered,motorMvtDataFiltered]
print("Features Data Set : ")
print(completeFeaturesDataList)

finalFeaturesEarlyFusionDataSet = reduce(lambda left,right: pd.merge(left,right,on='PATNO'), completeFeaturesDataList)
print("Early Fusion Feature Dataset : ")
print(finalFeaturesEarlyFusionDataSet)

finalLabelData = labelDataFiltered
trainingData1 = verbalDataFiltered
trainingData2 = semanticDataFiltered
trainingData3 = motorMvtDataFiltered

verbalDataList = [verbalDataFiltered,labelDataFiltered]
verbalDataSet = reduce(lambda left,right: pd.merge(left,right,on='PATNO'), verbalDataList)


semanticDataList = [semanticDataFiltered,labelDataFiltered]
semanticDataSet = reduce(lambda left,right: pd.merge(left,right,on='PATNO'), semanticDataList)


motorMvtDataList = [motorMvtDataFiltered,labelDataFiltered]
motorMvtDataSet = reduce(lambda left,right: pd.merge(left,right,on='PATNO'), motorMvtDataList)
'''
seaborn.pairplot(verbalDataSet, hue ='COMPLT')
seaborn.pairplot(semanticDataSet, hue ='COMPLT')
seaborn.pairplot(motorMvtDataSet, hue ='COMPLT')
'''

print("Total Labels for Each Class : ")
print(finalDataSet['COMPLT'].value_counts())
'''
======== Training Validation Testing Split For Early Fusion
'''

X = finalFeaturesEarlyFusionDataSet.drop(['PATNO'], axis=1)
y = finalLabelData.drop(['PATNO'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[2]
print("--------------(Decision Trees)-----------------")

decision_tree_clf = tree.DecisionTreeClassifier()
decision_tree_clf = decision_tree_clf.fit(X_train, y_train)
tree.plot_tree(decision_tree_clf)
dt_y_val_pred = decision_tree_clf.predict(X_val)
print("Val Accuracy:",metrics.accuracy_score(y_val, dt_y_val_pred))
print("F1 Score:",metrics.f1_score(y_val, dt_y_val_pred))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_val, dt_y_val_pred))

dt_y_test_pred = decision_tree_clf.predict(X_val)
#print("Test Accuracy:",metrics.accuracy_score(y_test, dt_y_test_pred))

# In[3]
print("---------------(SVM)----------------")

svmModel = LinearSVC()
svmModel.fit(X_train, y_train)
svm_y_val_pred = svmModel.predict(X_val)
print("Val Accuracy:",metrics.accuracy_score(y_val, svm_y_val_pred))
print("F1 Score:",metrics.f1_score(y_val, svm_y_val_pred))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_val, svm_y_val_pred))

svm_y_test_pred = svmModel.predict(X_val)
#print("Test Accuracy:",metrics.accuracy_score(y_test, svm_y_test_pred))


# In[4]


print("---------------(Logistic Regression)----------------")

logisticRegressionModel = LogisticRegression(random_state=0,max_iter=1000)
logisticRegressionModel.fit(X_train, y_train)
lr_y_val_pred = logisticRegressionModel.predict(X_val)
print("Val Accuracy:",metrics.accuracy_score(y_val, lr_y_val_pred))
print("F1 Score:",metrics.f1_score(y_val, lr_y_val_pred))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_val, lr_y_val_pred))

# In[6]
print("---------------(NN)----------------")

class NN(torch.nn.Module):
    def __init__(self,input_size, H, y_output):
        super(NN, self).__init__()
        
        self.hidden_size = H;
        self.softmax = torch.nn.Softmax()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,hidden_size=32,num_layers=1,batch_first=True,dropout=0.2)
        self.lstm2 = torch.nn.LSTM(input_size=32,hidden_size=16,num_layers=1,batch_first=True,dropout=0.2)
        self.output = torch.nn.Linear(5200, 6)
        
    def forward(self, input):
        hidden = self.initHidden()
        output , hidden = self.lstm1(input)
        output , hidden = self.lstm2(output)
        output = torch.flatten(output,start_dim=1)
        output = self.output(output)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class ThreeLayerMLP(torch.nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)
        self.sigmoid = torch.nn.LogSigmoid()
        self.softmax = torch.nn.Softmax()
        self.dropout = torch.nn.Dropout(0.25)
        self.outputShape = D_out
        self.weight_init()
        
    def weight_init(self):
        torch.nn.init.zeros_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear3.weight)
        torch.nn.init.zeros_(self.linear4.weight)
        
        torch.nn.init.ones_(self.linear1.bias)
        torch.nn.init.ones_(self.linear2.bias)
        torch.nn.init.ones_(self.linear3.bias)
        torch.nn.init.ones_(self.linear4.bias)

    def forward(self, x):
        x= x.cuda()
        h1_relu = self.linear1(x).clamp(min=0)
        #h1_relu = self.dropout(h1_relu)
        h2_relu = self.linear2(h1_relu).clamp(min=0)
        #h2_relu = self.dropout(h2_relu)
        # y_pred = 1/(1 + np.exp(-self.linear3(h2_relu)))
        h3_relu = self.linear3(h2_relu).clamp(min=0)
        #h3_relu = self.dropout(h3_relu)
        y_pred = self.linear4(h3_relu)
        y_pred = self.softmax(y_pred)
        return y_pred

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
 
params = {'batch_size': 50,
      'shuffle': True,
      'num_workers': 0}

training_set = TensorDataset(torch.tensor(X_train.values.tolist()), torch.tensor(y_train['COMPLT'].tolist()))
training_generator = torch.utils.data.DataLoader(training_set, **params)
binary_nn_model = ThreeLayerMLP(len(X_train.columns),8,16,8,2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(binary_nn_model.parameters(), lr=0.001)
epochErrors = []
epochF1Score = []
epochList = []
for t in range(185): 
        train_loss = 0.0
        binary_nn_model.train()
        binary_nn_model.float()
        binary_nn_model.to(device)
        for train_batch_review , train_batch_label in training_generator:
            train_batch_review , train_batch_label = train_batch_review.to(device) , train_batch_label.to(device)
            output = binary_nn_model(train_batch_review.float())
            loss = loss_fn(output, train_batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        prediction_list = []
        target_list = []
        params = {'batch_size': 50,
              'shuffle': True,
              'num_workers': 0}
        val_set = TensorDataset(torch.tensor(X_val.values.tolist()), torch.tensor(y_val['COMPLT'].tolist()))
        val_generator = torch.utils.data.DataLoader(val_set, **params)
        for val_batch_review , val_batch_label in val_generator:
            val_batch_review , val_batch_label = val_batch_review.to(device) , val_batch_label.to(device)
            output = binary_nn_model(val_batch_review.float())
            predicted = np.argmax(output.data.cpu(), axis=-1)
            prediction_list.extend(predicted.tolist())
            target_list.extend(val_batch_label.tolist())
        
        #print("Val Accuracy after Epoch",t,":",metrics.accuracy_score(target_list, prediction_list))
        epochErrors.append(1-metrics.accuracy_score(target_list, prediction_list))
        epochF1Score.append(metrics.f1_score(target_list, prediction_list))
        epochList.append(t)

fig, ax = plt.subplots(figsize=(100, 6))
plt.plot(epochList, epochErrors, label = "Val Error")
plt.plot(epochList, epochF1Score, label = "F1 Score")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.xticks(np.arange(min(epochList), max(epochList)+1, 5.0))
plt.legend()
plt.show()


prediction_list = []
target_list = []

for local_batch_review , local_batch_label in training_generator:
    local_batch_review , local_batch_label = local_batch_review.to(device) , local_batch_label.to(device)
    output = binary_nn_model(local_batch_review.float())
    predicted = np.argmax(output.data.cpu(), axis=-1)
    prediction_list.extend(predicted.tolist())
    target_list.extend(local_batch_label.tolist())

print("Train Accuracy:",metrics.accuracy_score(target_list, prediction_list))
print("F1 Score:",metrics.f1_score(target_list, prediction_list))
print("Confusion Matrix:")
print(metrics.confusion_matrix(target_list, prediction_list))

prediction_list = []
target_list = []
params = {'batch_size': 50,
      'shuffle': True,
      'num_workers': 0}
val_set = TensorDataset(torch.tensor(X_val.values.tolist()), torch.tensor(y_val['COMPLT'].tolist()))
val_generator = torch.utils.data.DataLoader(val_set, **params)
for val_batch_review , val_batch_label in val_generator:
    val_batch_review , val_batch_label = val_batch_review.to(device) , val_batch_label.to(device)
    output = binary_nn_model(val_batch_review.float())
    predicted = np.argmax(output.data.cpu(), axis=-1)
    prediction_list.extend(predicted.tolist())
    target_list.extend(val_batch_label.tolist())

print("Val Accuracy:",metrics.accuracy_score(target_list, prediction_list))
print("F1 Score:",metrics.f1_score(target_list, prediction_list))
print("Confusion Matrix:")
print(metrics.confusion_matrix(target_list, prediction_list))