import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import gensim
import time
import pickle
start_time = time.time()

#Loops each word through the list of categories and assigns the word the most similar category
def category_chooser(word,categories,w2vmodel):
	most_similar_category = len(categories)-1
	#best similarity chosen as a minimum cosine similarity needed for another category to be chosen
	best_similarity = 0.6
	for i in range(len(categories)-1):
		temp_similarity = w2vmodel.wv.similarity(word,categories[i])
		if temp_similarity > best_similarity: 
			best_similarity = temp_similarity
			most_similar_category = i
	return most_similar_category

def model_predictions(dataset,pos_model,neg_model,positive_target,negative_target):
	#Calculates the predictions 
	pos_probs = pos_model.predict_proba(dataset)[:,1]
	neg_probs = neg_model.predict_proba(dataset)[:,1]
	return pos_probs,neg_probs

#pulling the stop words from nltk
stopwords = list(set(stopwords.words('english')))
replacement_values = re.compile('[^a-zA-Z]') #Removing all non-alphanumeric characters from stop words

folder_location = r'C:\\Users\\Giada\\Documents\\Python files\\Twitter tweet positivity ranking\\'

#Reading a long list of categories from the dataset catagory file
with open(folder_location+'Dataset_Categories.txt','r') as file:
    categories = file.read().splitlines()
file.close()

#Pulling the test dataset 
dset_folder_location = r'C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\tweet-sentiment-extraction\\'
test_data = pd.read_csv(dset_folder_location + "train.csv",index_col="textID")
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

#unpickling the models
with open(folder_location+'pos_model.sav', 'rb') as pickle_file:
	pos_model = pickle.load(pickle_file)
with open(folder_location+'neg_model.sav', 'rb') as pickle_file:
	neg_model = pickle.load(pickle_file)

dataset_final = pd.DataFrame(columns=categories,index=range(len(test_data)))
w2vmodel = gensim.models.KeyedVectors.load(folder_location+'w2vmodel')

for sentence in range(len(test_data)):
	print("Process completed %.2f percent" % float(100*(sentence/len(test_data))))
	print("--- %.2f seconds ---" % (time.time() - start_time))
	print(test_data["text"][sentence])

	split_text = test_data["text"][sentence].split()
	for tweet_number in range(len(split_text)):
        #Removing all words that start with an @ or # as these are usernames and often several combined words as a hashtag
		if split_text[tweet_number][:1] == '@':
			split_text[tweet_number] = ''
		if split_text[tweet_number][:1] == '#':
			split_text[tweet_number] = ''

        #Removing non-alphabetical characters
		split_text[tweet_number] = replacement_values.sub('', split_text[tweet_number])

        #Removing weblinks which will start with http
		if split_text[tweet_number][:4] == 'http':
			split_text[tweet_number] = ''

        #Making all words lowercase
		split_text[tweet_number] = split_text[tweet_number].lower()

        #Removing stop words
		if split_text[tweet_number] in stopwords:
			split_text[tweet_number] = ''


    #Removing missing values left behind by strings of only non-alphabetical characters
	split_text = list(filter(None, split_text))

	#Setting all initial values to 0 in the final dset
	dataset_final.iloc[sentence,:] = 0
	#Loop through all of the words in the tweet
	for word_number in range(len(split_text)):
		word = split_text[word_number]
		#If statement to check the words are in the model vocab
		if word is not np.nan and word in w2vmodel.wv.vocab.keys():
			category_number = category_chooser(word=word,categories=categories,w2vmodel=w2vmodel)
			dataset_final.iloc[sentence,category_number] = dataset_final.iloc[sentence,category_number] + 1
	print(split_text)

dataset_final.to_csv(folder_location+'final_dset_train.csv',index=True)


pos_probs,neg_probs =	model_predictions(dataset=dataset_final
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

test_data.to_csv(folder_location+'Output_dset2.csv',index=True)