import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
	
folder_location = r'C:\\Users\\Giada\\Documents\\Python files\\Twitter tweet positivity ranking\\'

#Reading in the dataframe and creating seperate dataframes for the negative and positive statements
df_all = pd.read_csv(folder_location + 'modelling_dset.csv', index_col=0)
df_pos = df_all.loc[df_all['SCORE'] > -0.5]

Y = df_pos.pop("SCORE")
X = df_pos

dt = DecisionTreeClassifier(random_state=999)

dt_hyper = dict(max_depth = [50,55,60,65,70],
				min_samples_leaf=[20,25,30,35,40],
				max_features = ['sqrt','log2'],
				criterion = ['gini','entropy'])


dt_grid = GridSearchCV(estimator=dt,
					param_grid=dt_hyper,
					scoring='roc_auc',
					verbose=1,
					n_jobs=-1)

start_time=datetime.now()
dt_gs = dt_grid.fit(X,Y)
end_time = datetime.now()

print("runtime = ", str(start_time-end_time))
print("")
print(dt_gs.best_params_)

