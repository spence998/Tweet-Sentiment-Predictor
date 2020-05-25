import tweepy
import pandas as pd
import numpy as np
import json
import string
import re #package that checks if a string matches a pattern eg being alphanumeric
from nltk.corpus import stopwords #Package from natural language toolkit to get rid of stop words

def search_for_hashtags(stopwords, max_words, no_of_tweets, hashtag_phrase, output_dset):
    consumer_token = "O72v0C7nz9Ygfr5cyufGs42ZM"
    consumer_secret = "951kq45KHoJdu91zo7WBvvx7Vp7oJQSi7Ed1bYZXqTupRyXW7j"
    access_token = "1232356759009988609-9C5JCZ744UOgjsj86Gcwhq973VQ3YL"
    access_token_secret = "rvm24BbxFdSmwwrlmHgxIWMs0zhbp1NvDEbIqci8012fK"
    auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    #making a list 1 to max words for the column titles for the dataset
    column_titles = list(["word00"])
    for x in range(49):
        column_titles.append("word"+str(x+1).zfill(2))
    df = pd.DataFrame(columns = column_titles)

    #for each tweet matching our hashtags, write relevant info to the dataframe
    for tweet in tweepy.Cursor(api.search, q=hashtag_phrase+' -filter:retweets', lang="en", tweet_mode='extended').items(no_of_tweets):
    	#Splitting the chain of words up into a list of each individual word
        #Usually called "Tokenisation"
        words_temp_list = tweet.full_text.split()
        
        for x in range(len(words_temp_list)):
            #Removing all words that start with an @ or # as these are usernames and often several combined words as a hashtag
            if words_temp_list[x][:1] == '@':
                words_temp_list[x] = ''
            if words_temp_list[x][:1] == '#':
                words_temp_list[x] = ''

            #Removing non-alphabetical characters
            words_temp_list[x] = replacement_values.sub('', words_temp_list[x])

            #Removing weblinks which will start with http
            if words_temp_list[x][:4] == 'http':
                words_temp_list[x] = ''

            #Making all words lowercase
            words_temp_list[x] = words_temp_list[x].lower()

            #Removing stop words
            if words_temp_list[x] in stopwords:
                words_temp_list[x] = ''

        #Removing missing values left behind by strings of only non-alphabetical characters
        words_temp_list = list(filter(None, words_temp_list)) 

        #Adding null values onto the end of list where there are no more words
        for i in range(len(words_temp_list)+1,max_words + 1):
        	words_temp_list.append(np.nan)

        #Adding the column names next to each word so that it can be added to the dataframe
        words_list = [{"word"+str(i).zfill(2):words_temp_list[i] for i in range(max_words)}]
        df = df.append(words_list,ignore_index=True)
    output_dset = pd.concat([output_dset,df],ignore_index=True,sort=True)
    return output_dset



max_words = 50
no_of_tweets = 50
tweet_csv_file_location = r'C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\tweets_negative.csv'

stopwords = list(set(stopwords.words('english')))
#Removing all non-alphanumeric characters from stop words
replacement_values = re.compile('[^a-zA-Z]')
for x in range(len(stopwords)):
    stopwords[x] = replacement_values.sub('', stopwords[x])

df_accumulated = pd.read_csv(tweet_csv_file_location,low_memory=False)

#Reading a long list of "positive" hashtags from a file
with open('C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\hash_tag_file_negative.txt','r') as file:
    hashtag_phrase_list = file.read().splitlines()

#Running the data download function for all words in the list
for i in range(len(hashtag_phrase_list)):
    hashtag_phrase = hashtag_phrase_list[i]
    hashtag_phrase = "'{}'".format(hashtag_phrase)
    print(hashtag_phrase)
    df_accumulated = search_for_hashtags(
                        stopwords = stopwords,
                        max_words = max_words,
                        no_of_tweets = no_of_tweets,
                        hashtag_phrase=hashtag_phrase,
                        output_dset = df_accumulated
                        )

df_accumulated.drop_duplicates(keep='last', inplace=True)
df_accumulated.to_csv(tweet_csv_file_location,index=False)



