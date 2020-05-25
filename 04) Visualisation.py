#Code to plot the word2vec model
#This code is very slow if the word min count is low (~500 to get effective results)

import re
import matplotlib.pyplot as plt
import gensim
import numpy as np
from sklearn.manifold import TSNE

def tsne_plot(w2vmodel):

    #Putting the feature vectors and word labels for each word into a 1d array
    W2V_matrix = []
    labels = []
    for word in w2vmodel.wv.vocab:
        W2V_matrix.append(w2vmodel[word])
        labels.append(word)

    #TSNE converts the word2vec large matrix into 2 dimensions (if n_components=2) - coordinates[i][0=x,1=y] 
    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=2500, random_state=1)
    coordinates = tsne_model.fit_transform(W2V_matrix)    

    #Uses matlibplot to plot the word2vec
    plt.figure(figsize=(10, 10)) 
    for i in range(len(coordinates)):
        plt.scatter(coordinates[i][0],coordinates[i][1])
        plt.annotate(labels[i],
                     xy=(coordinates[i][0], coordinates[i][1]),
                     xytext=(5, 2),
                     textcoords='offset points')
    plt.show()
    
w2vmodel = gensim.models.KeyedVectors.load('C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\w2vmodel')
tsne_plot(w2vmodel)
