######################## Tweet Classifier ##########################

Description:

Project to build a model which is able to classify the sentiment of tweets.

The project consists of 10 codes which are run in order.

Files required to run the code and outputs for the code are stored in the
Additional Files used and output folder in the repository. These are to be
placed in the folder location specified in the code.

######################## Requirements ##########################

Requirements:

    Packages - Pandas               - Numpy
             - Scikit learn         - MatplotLib
             - NLTK                 - Gensim
             - Tweepy               - Pickle

##############################################################

Code description:
  
    1a-c) Scraping data from twitter and cleaning the data into a format that can
    be used by the word2vec. Due to the nature of the twitter of twitter API only 
    ~2000 tweets can be downloaded every 15 minutes. Therefore all sections were 
    needed to be run multiple times.
    
    2) Collating the tweets into one combined dataset
    
    3) Building of word2vec model. Hyperparameters have been optimised later on in
    the project using code 4/5 and a grid search.
    
    4-5) Visualisation and analysis of the word2vec model which are used to
    optimise the model.
    
    6) Creation of categories which will be used to group words based on individual words
    cosine similarity to each other. This is done by using the most frequent words which 
    are not "similar" to each other.
    
    7) Creation of modelling dataset created by categorising each word to a coresponding
    category made in code 06 using each words similarity.
    
    8) Testing of several models to determine the most predictive model to use. The 
    performance of the models is determined using gini in predicting positive and 
    negative sentiments. 
    
    Two logistic regression models were used, one for predicting positive and one 
    predicting negative sentiments. These models were chosen due to their relatively 
    high gini and low overfitting to the training data compared to the test data.
    
    9a-b) Optimising of hyperparameters for decision tree and random forest models using 
    a grid search.
    
    10a-c) Testing the model using a dataset downloaded from kaggle. The probability  on both the 
    positive and negative are used in conjunction with a set of score rules to predict the
    outcome of the model. 
    
    The final predictions can be found in the file Outcome.csv
    
    The model predicts the sentiments with an accuracy of 54%.

##############################################################
