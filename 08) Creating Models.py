import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics

import matplotlib.pyplot as plt 
import pickle

def data_split(df, target):
	Y = df.pop("SCORE")
	X = df
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state=999)

	print("--------------------------------------------------------")
	print("-------------------" + target + "------------------")
	print("--------------------------------------------------------")
	print("Training data length:" + str(len(Y_train)))
	print("Test data length:" + str(len(Y_test)))
	print("Ratio: " + str(len(Y_test) / (len(Y_test)+len(Y_train))))

	total_target_statements = sum(Y_test) + sum(Y_train)
	print("Total target Statements = " + str(total_target_statements))
	print("Train target Statements = " + str(sum(Y_train)))
	print("Test target Statements = " + str(sum(Y_test)))
	print("Ratio: " + str(sum(Y_test) / (sum(Y_test)+sum(Y_train))))
	return X_train,X_test,Y_train,Y_test
	

def plot_roc_curve(y_true, score_names):
	#compute gini for each model
	for score, label in score_names:	
		fpr,tpr, _ = roc_curve(y_true,y_score=score,drop_intermediate=False)
		AUC = roc_auc_score(y_true,score)
		gini = 2*AUC - 1
		label = "gini = %.5f %s" % (gini, label)
		print(gini)
	return fpr,tpr

def logistical_regression_model(X_train,X_test,Y_train,Y_test,target):
	#Training the logistical regression models and testing them 
	logreg = LogisticRegression(random_state = 999)
	logreg.fit(X_train, Y_train)

	train_probs = logreg.predict_proba(X_train)[:,1]
	test_probs = logreg.predict_proba(X_test)[:,1]

	#Both of the train and test ginis are the same so only one of each will be plotted
	FPR,TPR = plot_roc_curve(Y_train,[(train_probs,'Test: Logistic Regression Model: '+target)])
	FPR,TPR = plot_roc_curve(Y_test,[(test_probs,'Test: Logistic Regression Model: '+target)])

	return logreg, FPR, TPR

def random_forest_model(X_train,X_test,Y_train,Y_test,target):
	#Random Forest hyper perameter chosen by the grid search in program 09a
	rf = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=22,max_features='sqrt',min_samples_leaf=6,random_state=999)
	rf.fit(X_train, Y_train)

	train_probs = rf.predict_proba(X_train)[:,1]
	test_probs = rf.predict_proba(X_test)[:,1]

	#Both of the train and test ginis are the same so only one of each will be plotted
	FPR,TPR = plot_roc_curve(Y_train,[(train_probs,'Test: Random Forest Model: '+target)])
	FPR,TPR = plot_roc_curve(Y_test,[(test_probs,'Test: Random Forest Model: '+target)])
	
	return rf, FPR, TPR

def decistion_tree_model(X_train,X_test,Y_train,Y_test,target):
	#Decision tree hyper perameter chosen by the grid search in program 09b
	dt = DecisionTreeClassifier(criterion='gini',max_depth=60,max_features='log2',min_samples_leaf=30,random_state=999)
	dt.fit(X_train, Y_train)

	train_probs = dt.predict_proba(X_train)[:,1]
	test_probs = dt.predict_proba(X_test)[:,1]

	#Both of the train and test ginis are the same so only one of each will be plotted
	FPR,TPR = plot_roc_curve(Y_train,[(train_probs,'Test: Decision Tree Model: '+target)])
	FPR,TPR = plot_roc_curve(Y_test,[(test_probs,'Test: Decision Tree Model: '+target)])
	
	return dt, FPR, TPR


folder_location = r'C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\'

#Reading in the dataframe and creating seperate dataframes for the negative and positive statements
df_all = pd.read_csv(folder_location + 'modelling_dset.csv', index_col=0)
df_pos = df_all.loc[df_all['SCORE'] > -0.5]
df_neg = df_all.loc[df_all['SCORE'] < 0.5]
df_neg['SCORE'] = abs(df_neg['SCORE'])

#Running the functions to create the models for positive and negative models
X_train,X_test,Y_train,Y_test = data_split(df_pos,"Positive Statements")
pos_model_LR,pos_FPR_LR,pos_TPR_LR = logistical_regression_model(X_train,X_test,Y_train,Y_test,target="positive_model")
pos_model_RF,pos_FPR_RF,pos_TPR_RF = random_forest_model(X_train,X_test,Y_train,Y_test,target="positive_model")
pos_model_DT,pos_FPR_DT,pos_TPR_DT = decistion_tree_model(X_train,X_test,Y_train,Y_test,target="positive_model")

X_train,X_test,Y_train,Y_test = data_split(df_neg,"Negative Statements")
neg_model_LR,neg_FPR_LR,neg_TPR_LR = logistical_regression_model(X_train,X_test,Y_train,Y_test,target="negative_model")
neg_model_RF,neg_FPR_RF,neg_TPR_RF = random_forest_model(X_train,X_test,Y_train,Y_test,target="negative_model")
neg_model_DT,neg_FPR_DT,neg_TPR_DT = decistion_tree_model(X_train,X_test,Y_train,Y_test,target="negative_model")

#Saving models using pickle
pickle.dump(pos_model_LR, open(folder_location+'pos_model.sav', 'wb'))
pickle.dump(neg_model_LR, open(folder_location+'neg_model.sav', 'wb'))


#plotting the gini curves
plt.figure(figsize=(11,11))
plt.plot(pos_FPR_LR,pos_TPR_LR,label="pos_model_LR")
plt.plot(pos_FPR_RF,pos_TPR_RF,label="pos_model_RF")
plt.plot(pos_FPR_DT,pos_TPR_DT,label="pos_model_DT")
plt.plot(neg_FPR_LR,neg_TPR_LR,label="neg_model_LR")
plt.plot(neg_FPR_RF,neg_TPR_RF,label="neg_model_RF")
plt.plot(neg_FPR_DT,neg_TPR_DT,label="neg_model_DT")
plt.plot((0,1),(0,1),label="Straight line")
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


