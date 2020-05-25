import gensim
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

w2vmodel = gensim.models.KeyedVectors.load('C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\w2vmodel')

#Loading all the values form the word2vec into a dictionary and sorting them based on their frequency
w2c = dict()
for item in w2vmodel.wv.vocab:
    w2c[item]=w2vmodel.wv.vocab[item].count
w2cSorted=dict(sorted(w2c.items(), key = lambda x : x[1],reverse=True))

#Used as a word frequency cut-off if necessary
"""
for key, value in dict(w2cSorted).items():
    if value < 100:
        del w2cSorted[key] 
words = w2cSorted.keys()
"""

#Changing the format of the frequency of words from a dictionary to a list to be used in the graph
dict_frequency = w2cSorted.values()
frequency = list(dict_frequency)

#Creating a scale 1,2,3... as a x-axis for the graph representing the "word number"
word_number = list()
for i in range(len(frequency)):
	word_number.append(len(frequency) - i)

#Plotting the graph
fig = plt.plot(word_number,frequency)
fig = plt.yscale(value="log")
fig = plt.xlabel("word number")
fig = plt.ylabel("frequecy")
plt.show()
