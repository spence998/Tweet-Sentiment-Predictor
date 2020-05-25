import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
	
folder_location = r'C:\\Users\\Giada\\Documents\\Python files\\Twitter tweet positivity ranking\\'

#Reading in the dataframe and creating seperate dataframes for the negative and positive statements
df_all = pd.read_csv(folder_location + 'final_dset.csv', index_col=0)
df_pos = df_all.loc[df_all['SCORE'] > -0.5]

Y = df_pos.pop("SCORE")
X = df_pos

rf = RandomForestClassifier(random_state=999)

rf_hyper = dict(n_estimators=[200],
				max_depth = [16,18,20,22,24],
				min_samples_leaf=[3,4,5,6,7,8],
				max_features = ['sqrt','log2'],
				criterion = ['gini','entropy'])


rf_grid = GridSearchCV(estimator=rf,
					param_grid=rf_hyper,
					scoring='roc_auc',
					verbose=1,
					n_jobs=-1)

start_time=datetime.now()
rf_gs = rf_grid.fit(X,Y)
end_time = datetime.now()

print("runtime = ", str(start_time-end_time))
print("")
print(rf_gs.best_params_)

