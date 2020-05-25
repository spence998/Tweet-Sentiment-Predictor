import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import time
import pickle
start_time = time.time()

def model_predictions(dataset,pos_model,neg_model,positive_target,negative_target):
	pos_probs = pos_model.predict_proba(dataset)[:,1]
	neg_probs = neg_model.predict_proba(dataset)[:,1]

	return pos_probs,neg_probs

folder_location = r'C:\\Users\\Giada\\Documents\\Python files\\Twitter tweet positivity ranking\\'

#Pulling the test dataset 
dset_folder_location = r'C:\\Users\\Giada\\Documents\\Python files\\Twitter tweet positivity ranking\\tweet-sentiment-extraction\\'
test_data = pd.read_csv(dset_folder_location + "test.csv",index_col="textID")
text_list = test_data["text"]
test_target = list(test_data["sentiment"])

#Positive target list
test_target_pos = []
for target in range(len(test_target)):
	if test_target[target] == "positive":
		test_target_pos.append(1)
	else:
		test_target_pos.append(0)

#Negative target list
test_target_neg = []
for target in range(len(test_target)):
	if test_target[target] == "negative":
		test_target_neg.append(1)
	else:
		test_target_neg.append(0)

#unpickling the predictive models
with open(folder_location+'pos_model.sav', 'rb') as pickle_file:
	pos_model = pickle.load(pickle_file)
with open(folder_location+'neg_model.sav', 'rb') as pickle_file:
	neg_model = pickle.load(pickle_file)

modelling_dset = pd.read_csv(folder_location+'modelling_kaggle_dset_test.csv', low_memory=False,index_col=0)

pos_probs, neg_probs = 	model_predictions(dataset=modelling_dset
						,pos_model=pos_model
						,neg_model=neg_model
						,positive_target=test_target_pos
						,negative_target=test_target_neg)
	
test_data["pos_probs"] = pos_probs
test_data["neg_probs"] = neg_probs

#-----------------------------------------------------------
#-HERE ARE A SERIES OF IF STATEMENTS TO DECIDE THE OUTCOME--
#-THIS HAS BEEN OPTIMISED IN CODE 10b-----------------------
#-----------------------------------------------------------
pos_cutoff = 0.39
neg_cutoff = 0.57
#-----------------------------------------------------------
test_data.loc[(test_data["pos_probs"] > pos_cutoff) & (test_data["neg_probs"] > neg_cutoff), "OUTCOME"] = "BOTH"
test_data.loc[(test_data["pos_probs"] > pos_cutoff) & (test_data["neg_probs"] < neg_cutoff), "OUTCOME"] = "positive"
test_data.loc[(test_data["pos_probs"] < pos_cutoff) & (test_data["neg_probs"] > neg_cutoff), "OUTCOME"] = "negative"
test_data.loc[(test_data["pos_probs"] < pos_cutoff) & (test_data["neg_probs"] < neg_cutoff), "OUTCOME"] = "neutral"

test_data.loc[(test_data["OUTCOME"] == "BOTH") & (test_data["pos_probs"] > test_data["neg_probs"]), "OUTCOME"] = "positive"
test_data.loc[(test_data["OUTCOME"] == "BOTH") & (test_data["pos_probs"] < test_data["neg_probs"]), "OUTCOME"] = "negative"

test_data.loc[test_data["OUTCOME"] == test_data["sentiment"], "ACCURACY"] = 1
test_data.loc[test_data["OUTCOME"] != test_data["sentiment"], "ACCURACY"] = 0

#Calculating the accuracy of the prediction
total_correct_predictions = test_data["ACCURACY"].sum()
accuracy = total_correct_predictions/len(test_data)
print(accuracy)

test_data.to_csv(folder_location+'Output_dset.csv',index=True)
