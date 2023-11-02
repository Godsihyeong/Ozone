import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


df = pd.read_csv('데이터최종.csv')
df.set_index(pd.to_datetime(df[df.columns[0]]), inplace=True)
df.drop(df.columns[0], axis=1, inplace=True)
df = df.rename(columns={'오전 이산회질소 농도 평균':'오전 이산화질소 농도 평균'})
df = df.interpolate()
df.reset_index(inplace=True)
df
df.drop(df.columns[0], axis = 1, inplace = True)
df
X = df[[df.columns[0], df.columns[1], df.columns[3], df.columns[4],df.columns[5]]]
y = df[df.columns[6]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=202)

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
########### 표준화
########### 표준화
########### 표준화

scaler = StandardScaler()
scaler.fit(X)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
knn_model_s = KNeighborsRegressor()
knn_model_s.fit(X_train_s, y_train)
y_pred_s = knn_model_s.predict(X_test_s)
r2_score(y_pred_s,y_test)
grid_params = {
    'n_neighbors' : list(range(1,20)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski']
}
gs_s = GridSearchCV(knn_model_s, grid_params, cv = 10)
gs_s.fit(X_train_s, y_train)
print("Best Parameters : ", gs_s.best_params_)
print("Best Score : ", gs_s.best_score_)
print("Best Test Score : ", gs_s.score(X_test_s, y_test))
knn_model_bests = KNeighborsRegressor(metric ='manhattan', n_neighbors = 7, weights = 'distance')
knn_model_bests.fit(X_train_s, y_train)
y_pred_bests = knn_model_bests.predict(X_test_s)
r2_score(y_pred_bests,y_test)
y_test.reset_index()['오후 오존 농도']
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(y_test.reset_index()['오후 오존 농도'], y_pred_bests, s = 3)

x = np.linspace(0,0.10,5)
y = x

plt.plot(x,y, 'r-')


########### preprocessing 안함
########### preprocessing 안함
########### preprocessing 안함

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
r2_score(y_pred,y_test)
gs = GridSearchCV(knn_model, grid_params, cv = 10)
gs.fit(X_train, y_train)
print("Best Parameters : ", gs.best_params_)
print("Best Score : ", gs.best_score_)
print("Best Test Score : ", gs.score(X_test, y_test))
############# 정규화
############# 정규화
############# 정규화

mscaler = MinMaxScaler()
mscaler.fit(X)
X_train_m = mscaler.transform(X_train)
X_test_m = mscaler.transform(X_test)

knn_model_m = KNeighborsRegressor()
knn_model_m.fit(X_train_m, y_train)
y_pred_m = knn_model_m.predict(X_test_m)
r2_score(y_pred_m,y_test)
from sklearn import metrics
import numpy as np

np.sqrt(metrics.mean_squared_error(y_pred_m, y_test))
gs_m = GridSearchCV(knn_model_m, grid_params, cv = 10)
gs_m.fit(X_train_m, y_train)
print("Best Parameters : ", gs_m.best_params_)
print("Best Score : ", gs_m.best_score_)
print("Best Test Score : ", gs_m.score(X_test_m, y_test))
knn_model_bestm = KNeighborsRegressor(metric ='manhattan', n_neighbors = 5, weights = 'distance')
knn_model_bestm.fit(X_train_m, y_train)
y_pred_bestm = knn_model_bestm.predict(X_test_m)
r2_score(y_pred_bestm,y_test)
from sklearn import metrics
import numpy as np

np.sqrt(metrics.mean_squared_error(y_pred_bestm, y_test))
############# RobustScaler
############# RobustScaler
############# RobustScaler

from sklearn.preprocessing import RobustScaler

rscaler = RobustScaler()
rscaler.fit(X)
X_train_r = rscaler.transform(X_train)
X_test_r = rscaler.transform(X_test)


knn_model_r = KNeighborsRegressor()
knn_model_r.fit(X_train_r, y_train)
y_pred_r = knn_model_r.predict(X_test_r)
r2_score(y_pred_r,y_test)

gs_r = GridSearchCV(knn_model_r, grid_params, cv = 10)
gs_r.fit(X_train_r, y_train)
print("Best Parameters : ", gs_r.best_params_)
print("Best Score : ", gs_r.best_score_)
print("Best Test Score : ", gs_r.score(X_test_r, y_test))
knn_model_bestr = KNeighborsRegressor(metric ='manhattan', n_neighbors = 8, weights = 'distance')
knn_model_bestr.fit(X_train_r, y_train)
y_pred_bestr = knn_model_bestr.predict(X_test_r)
r2_score(y_pred_bestr,y_test)
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
####### SVR
####### SVR
####### SVR

from sklearn.svm import SVR

clf = SVR(kernel='poly', epsilon=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(r2_score(y_pred, y_test))
print(mean_squared_error(y_test, y_pred))
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

#### RandomForestRegressor
#### RandomForestRegressor
#### RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

clf = RandomForestRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(r2_score(y_pred, y_test))
print(mean_squared_error(y_test, y_pred))
print(clf.score(X_train, y_train))
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager, rc
    
font_path = "C:\Windows\Fonts\malgun.ttf"
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

importance = clf.feature_importances_
importances = pd.Series(importance, index = [df.columns[0],df.columns[1],df.columns[3],df.columns[4],df.columns[5]])
top20 = importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
sns.barplot(x=top20, y=top20.index)
plt.title('변수 중요도 순위')
plt.tight_layout()
plt.show()
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
#### AdaBoost
#### AdaBoost
#### AdaBoost

from sklearn.ensemble import AdaBoostRegressor

clf = AdaBoostRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(r2_score(y_pred, y_test))
print(mean_squared_error(y_test, y_pred))
##### ANN model
##### ANN model
##### ANN model


import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
X = df[[df.columns[0], df.columns[1], df.columns[3], df.columns[4],df.columns[5]]]
y = df[df.columns[6]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=202)

########### 표준화
########### 표준화
########### 표준화

scaler = StandardScaler()
scaler.fit(X)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

############# 정규화
############# 정규화
############# 정규화

mscaler = MinMaxScaler()
mscaler.fit(X)
X_train_m = mscaler.transform(X_train)
X_test_m = mscaler.transform(X_test)


# kernel_initializer='random_normal', activation = 'relu'
model = Sequential()
model.add(layers.Dense(units=1, input_dim = 5, kernel_initializer='random_normal', activation = 'relu'))
model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])
model.fit(X_train_m, y_train, epochs = 50)
prediction = model.predict(X_test_m)
r2_score(y_test, prediction)
from sklearn import metrics
import numpy as np
np.sqrt(metrics.mean_squared_error(y_test, prediction))
from ann_visualizer.visualize import ann_viz

ann_viz(model, view=True, filename='test1')
model.summary()
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


