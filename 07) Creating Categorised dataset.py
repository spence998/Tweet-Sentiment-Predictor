import pandas as pd
import numpy as np
import gensim
import time
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

########################################################################################
#	THIS CODE RUNS THROUGH EACH WORD OF EACH TWEET AND ASSIGNS IT TO A CATEGORY SET    #
#	IN THE PREVIOUS CODE 06. 
########################################################################################
#	THE CODE TAKES A LONG TIME TO RUN: ~45mins										   #
########################################################################################

folder_location ='C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\'

#Reading a long list of categories from the dataset catagory file
with open(folder_location+'Dataset_Categories.txt','r') as file:
    categories = file.read().splitlines()
file.close()

df = pd.read_csv(folder_location+'tweets_Combined.csv', low_memory=False)
SCORE = df.pop("SCORE")
dataset_final = pd.DataFrame(columns=categories,index=range(len(df)))
w2vmodel = gensim.models.KeyedVectors.load(folder_location+'w2vmodel')

#Loop to go through each of the tweets
for tweet_number in range(len(df)): 
	print("Process completed %.2f percent" % float(100*(tweet_number/len(df))))
	print("--- %.2f seconds ---" % (time.time() - start_time))
	#Initial setting of all categories to 0 
	dataset_final.iloc[tweet_number,:] = 0

	#Loop through all of the words in the tweet
	for word_number in range(50):
		word = df.iloc[tweet_number,word_number]
		#If statement to check the words are in the model vocab
		if word is not np.nan and word in w2vmodel.wv.vocab.keys():
			category_number = category_chooser(word=word,categories=categories,w2vmodel=w2vmodel)
			dataset_final.iloc[tweet_number,category_number] = dataset_final.iloc[tweet_number,category_number] + 1

dataset_final["SCORE"] = SCORE

dataset_final.to_csv(folder_location+'modelling_dset.csv',index=True)




