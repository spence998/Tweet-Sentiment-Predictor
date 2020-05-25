import gensim
import pandas as pd
import numpy as np

folder_location = r'C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\'
df = pd.read_csv(folder_location+'Tweets_Combined.csv', low_memory=False)

#Transforming the dataset of words in a list of tweets where each tweet is comprised of a 
#list of the tweets individual words.
df.pop('SCORE')
Row_list = []
for i in range((df.shape[0])): 
	cleanedList = [x for x in list(df.iloc[i, :]) if str(x) != 'nan']
	Row_list.append(list(cleanedList))

#Training the word2vec model 
w2vmodel = gensim.models.Word2Vec(window=2,
							   min_count=10,
							   sg=0,
							   alpha=0.025,
							   size=100,
							   )
w2vmodel.build_vocab(Row_list)  # prepare the model vocabulary
w2vmodel.train(sentences=Row_list, total_examples=len(Row_list), epochs=w2vmodel.epochs)
#w2vmodel.save(folder_location+'w2vmodel')

#For checks for testing the strength of the word2vec
print(w2vmodel)
print("")
print(w2vmodel.wv.most_similar("brilliant",topn=10))
print(w2vmodel.wv.doesnt_match(["good","great","fantastic","bad"]))

