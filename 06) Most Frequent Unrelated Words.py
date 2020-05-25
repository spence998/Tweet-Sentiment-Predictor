import gensim
import pandas as pd 
import numpy as np 

#If two words have a similarity of more than the limit then this function removes the less frequent word from the list
def similar_word_remover(model,word_list,similarity_limit):
	new_list = []
	for i in range(len(word_list)):
		for j in range(len(word_list)):
			if model.wv.similarity(word_list[i],word_list[j]) > similarity_limit and model.wv.similarity(word_list[i],word_list[j]) != 1:
				if i>j:
					new_list.append(word_list[i])
				else:
					new_list.append(word_list[j])
	new_list = list(dict.fromkeys(new_list))

	final_list = word_list
	for word in new_list:
		final_list.remove(word)
	return final_list				

#Function to create a similarity matrix in a dataframe and output it to a csv file
def similarity_matrix(model, word_list,folder_location):
	df = pd.DataFrame(columns = word_list, index=word_list)
	for i in range(len(word_list)):
		for j in range(len(word_list)):
			df.iloc[i,j] = w2vmodel.wv.similarity(word_list[i],word_list[j])
	df.to_csv(folder_location+'similarity_matrix_data.csv',index=True)

########################################################################################
#	THIS CODE CREATES A LIST OF CATEGORIES FOR EACH WORD IN THE VOCAB TO BE 		   #
#	CATEGORISED INTO DEPENDING ON ITS SIMILARITY TO THE OTHER WORDS IN CATEGORY		   #
########################################################################################
#	HOW MANY OF THE MOST FREQUENT WORDS DO YOU WANT THE CATEGORISES TO BE CHOSEN FROM? #
number_of_catagories = 250															   #			
#	UPPER LIMIT FOR WORD SIMILARITY 												   #
similarity_limit = 0.75																   #
########################################################################################

folder_location = 'C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\'
w2vmodel = gensim.models.KeyedVectors.load(folder_location+'w2vmodel')

#Loading all the values form the word2vec into a dictionary and sorting them based on their frequency
w2c = dict()
for item in w2vmodel.wv.vocab:
    w2c[item]=w2vmodel.wv.vocab[item].count
w2cSorted=dict(sorted(w2c.items(), key = lambda x : x[1],reverse=True))

#Creating a list of the most frequent words
dict_words = w2cSorted.keys()
ordered_words = list(dict_words)
most_frequent_words = ordered_words[0:number_of_catagories]

new_word_list = similar_word_remover(model=w2vmodel, word_list=most_frequent_words,similarity_limit=similarity_limit)
similarity_matrix(model=w2vmodel, word_list=new_word_list,folder_location=folder_location)

#Writing the final dataset categories to a txt file
with open(folder_location + 'Dataset_Categories.txt','w') as file:
    for word in new_word_list:
   		file.write(word + "\n")
    	file.write("OTHER" + "\n")
file.close()
