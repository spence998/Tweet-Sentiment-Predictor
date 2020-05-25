import pandas as pd
import numpy as np

#Function to load data and remove tweets with less than 5 words
def Loading_Data(csv_file_location):
	df = pd.read_csv(csv_file_location,low_memory=False)
	df_copy = df.copy()
	for i in range(len(df)):
		if df.iloc[i,5] is np.nan:
			df_copy = df_copy.drop([i], axis=0)
	return df_copy

folder_location = r'C:\\Users\\Spencer\\Documents\\Python files\\Twitter tweet positivity ranking\\'

positive_df = Loading_Data	(csv_file_location = folder_location + 'tweets_positive.csv')
negative_df = Loading_Data	(csv_file_location = folder_location + 'tweets_negative.csv')
neutral_df = Loading_Data	(csv_file_location = folder_location + 'tweets_neutral.csv')

#Setting a target score representing the sentiment of each tweet
positive_df["SCORE"] = 1
negative_df["SCORE"] = -1
neutral_df["SCORE"] = 0

total_dset = positive_df.append(negative_df,ignore_index=True)
total_dset = total_dset.append(neutral_df,ignore_index=True)

total_dset.drop_duplicates(keep='last', inplace=True)

total_dset.to_csv(folder_location + 'Tweets_Combined.csv',index=False)